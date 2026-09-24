# Yerel Agent Projesi - Kod Rehberi

Bu belge, projedeki uygulama kodunu, arayüzü, testleri ve çalışma dosyalarını tek tek tanıtır. Açıklamalar mevcut kaynak koduna göredir. Bir bileşenin nerede kullanıldığını görmek için **Okuma sırası** bölümünden başlayıp dosya yolunu editörde açabilirsin.

## Sistemin kısa haritası

```text
app/api.py veya app/main.py
        |
        v
app/service/tasks.py ---- görev kuyruğu, sohbet ve onay
        |
        v
app/service/runtime.py -- model, araçlar ve çalışma modunu kurar
        |
        +--> AutoModeRouter --> chat / single / supervisor / plan
        |                           |
        |                           +--> Supervisor --> uzman worker --> tools
        |                                                       |
        |                                                       +--> Reviewer
        +--> graph modu --> LangGraph üzerinden Supervisor/worker/Reviewer
```

Varsayılan akışta AutoModeRouter genel modu seçer. Supervisor modu seçildiyse Supervisor LLM'i uzman worker seçer ve işin tamamlanıp tamamlanmadığına karar verir. `graph` modu ayrıca seçilirse aynı temel görev rollerini LangGraph `StateGraph` içinde çalıştırır. Bu yüzden Supervisor akışı ile LangGraph aynı kavram değildir.

## Okuma sırası

Projeyi öğrenirken bu sırayla aç:

1. `app/api.py` veya CLI kullanıyorsan `app/main.py`
2. `app/service/tasks.py`
3. `app/service/runtime.py`
4. `app/orchestration/auto_mode.py`
5. `app/agents/supervisor.py`
6. `app/agents/workers.py`, sonra `app/agents/single.py`
7. `app/tools/base.py`, `app/tools/registry.py`, sonra gereken tool dosyası
8. `app/agents/reviewer.py`
9. `app/orchestration/graph.py`
10. `app/orchestration/state.py`, `app/llm/` ve bellek/gözlemleme klasörleri

## Uygulama kodu: `app/`

### Giriş ve kullanıcı arayüzü

| Dosya | Ne yapıyor? |
|---|---|
| `app/__init__.py` | `app` Python paketini tanımlar. Uygulama davranışı yok denecek kadar azdır. |
| `app/main.py` | Terminal/CLI giriş noktasıdır. Etkileşimli sohbeti, agent çalıştırmayı ve bellek komutlarını başlatır. `python -m app.main` buraya gelir. |
| `app/api.py` | FastAPI yerel web uygulamasının giriş noktasıdır. Arayüz dosyalarını sunar; görev başlatma, görev durumu/olayları, sohbet geçmişi ve kalıcı bellek gibi HTTP uçlarını tanımlar. |
| `app/ui/index.html` | Web arayüzünün HTML iskeleti: sohbet alanı, çalışma modu seçimi, görev akışı ve bellek bölümü. |
| `app/ui/app.js` | Tarayıcı davranışı: API çağrıları, sohbet mesajları, görev durumlarının canlı alınması, onay pencereleri ve akış görselleştirmesi. |
| `app/ui/style.css` | Arayüzün renk, yerleşim, düğüm/ok ve mobil görünüm stilleri. |

### Görev servisi ve çalışma zamanı

| Dosya | Ne yapıyor? |
|---|---|
| `app/service/tasks.py` | HTTP katmanı ile agent runtime arasındaki görev yöneticisidir. Görev kuyruğu, görev snapshot'ı, ilerleme durumu ve tarayıcıdan gelen araç onaylarını yönetir. |
| `app/service/runtime.py` | Çalışma zamanını kurar: workspace, tool registry, yerel LLM client, rol bazlı model sarmalayıcıları, worker'lar, reviewer ve seçilen orchestration modu. Modların nereden ayrıldığını anlamak için en önemli dosyalardan biridir. |
| `app/service/conversations.py` | Sohbet mesajlarının konuşma kimliğiyle saklanmasını ve yeniden yüklenmesini sağlar. SQLite ve bellek içi saklama seçenekleri bulunur. |
| `app/service/__init__.py` | Service paketinin işaret dosyası/açıklaması. |

### Agentlar

| Dosya | Ne yapıyor? |
|---|---|
| `app/agents/chat.py` | Araç veya plan gerektirmeyen doğrudan Türkçe sohbet yolu. Geçmişi modele ekler; belirli isim hatırlama, selamlaşma ve asistan kimliği cevaplarını deterministik verir. İçindeki `automatic_mode` basit kural tabanlı sınıflandırıcıdır; LLM kullanan mode router ile aynı şey değildir. |
| `app/agents/single.py` | Bir agent'ın kendi başına sınırlı bir döngüde çalışması: system prompt ve izinli tool şemalarıyla LLM'den tool/final kararı alır, aracı çağırır, cevabı toplar. Tekrarlanan tool hataları ve adım/çağrı limitleri de burada ele alınır. |
| `app/agents/workers.py` | Uzman worker adlarını, açıklamalarını, tam role talimatlarını ve tool allowlist'lerini tanımlar. `build_workers` her uzmanı SingleAgent tabanında üretir. |
| `app/agents/supervisor.py` | Ana supervisor karar döngüsüdür. LLM'den delege etme veya son yanıt kararı alır; seçilen worker'ı çalıştırır; sonuçları state'e ekler; gerektiğinde görevi parçalara ayırıp devam eder. Planlı görev yürütme ve retry limitleri de burada bulunur. |
| `app/agents/reviewer.py` | Worker çıktısını kanıt üzerinden inceler. Yazılmış dosyayı yeniden okuma, Python sözdizimi kontrolü, CSV ölçümlerini yeniden hesaplama, testleri tekrar çalıştırma ve araştırma kaynaklarının iddiaları destekleyip desteklemediğini kontrol etme gibi işler yapar. |
| `app/agents/__init__.py` | Agent paketinin işaret dosyası/açıklaması. |

### Yönlendirme, planlama ve görev state'i

| Dosya | Ne yapıyor? |
|---|---|
| `app/orchestration/auto_mode.py` | Otomatik modda `chat`, `single`, `supervisor` veya `plan` seçer. Önce bazı kesin kuralları uygular; diğer isteklerde LLM'den JSON karar ister, güven eşiği ve koruyucu kurallarla seçimi doğrular. |
| `app/orchestration/router.py` | Ayrı `router` modudur. Tek ve bağımsız bir iş için LLM'den worker önerisi alır. Düşük güven, çok adımlı istek, geçmiş veya gerekli inceleme varsa supervisor'a yönlendirir. |
| `app/orchestration/planner.py` | Çok adımlı görevleri 1-4 sıralı/bağımlı iş olarak JSON plana çevirir. Plan yalnızca uygun agent adlarını kullanabilir. Supervisor planı yürütür. |
| `app/orchestration/graph.py` | İsteğe bağlı LangGraph yoludur. `StateGraph` düğümleri Supervisor, worker ve Reviewer çağrılarını açık koşullu kenarlarla bağlar. Varsayılan otomatik yolun kendisi değildir. |
| `app/orchestration/state.py` | Görev boyunca taşınan Pydantic veri sözleşmeleri: tool sonucu/kaydı, worker çıktısı, plan, review sonucu, route kararı ve genel AgentState. |
| `app/orchestration/context.py` | Modele veya workera gönderilen bağlamı sınırlar; ilgili görev/önceki sonuçları seçer. Çok uzun geçmişin bağlam penceresini taşırmasını önlemeye çalışır. |
| `app/orchestration/evidence.py` | Başarılı Wikipedia/web tool kanıtlarına göre final araştırma yanıtını düzeltir; kaynakta olmayan iddiayı/URL'yi gerçekmiş gibi sunmayı engeller. |
| `app/orchestration/__init__.py` | Orchestration paketinin işaret dosyası/açıklaması. |

### LLM bağlantısı ve yapılandırılmış kararlar

| Dosya | Ne yapıyor? |
|---|---|
| `app/llm/client.py` | Agent kodunun kullandığı LLM arayüzü ve yerel GGUF/llama.cpp uygulaması. Modeli açma, chat çağrısı, JSON schema geçme, zaman aşımı ve backend hatalarını düzenler. |
| `app/llm/schemas.py` | ChatMessage, model yanıtı ve token kullanımı gibi ortak veri şekilleri. |
| `app/llm/structured.py` | LLM'nin tool kullan/final yanıt kararlarını JSON/Pydantic şemasına göre ayrıştırır. Geçersiz çıktı için sınırlı düzeltme denemeleri yapar. |
| `app/llm/strategy.py` | Roller için model seçme arayüzü. Mevcut strateji, tüm rolleri tek yerel model instance'ına bağlar; rollerin davranış farkını prompt ve izinler oluşturur. |
| `app/llm/__init__.py` | LLM paketinin işaret dosyası/açıklaması. |

### Tool altyapısı ve gerçek tool'lar

| Dosya | Ne yapıyor? |
|---|---|
| `app/tools/base.py` | Bütün tool'ların ortak girdi/çıktı sözleşmesi (`ToolSpec`, `ToolResult`, `BaseTool`). Pydantic girdiler tool sınırında doğrulanır. |
| `app/tools/registry.py` | Tool'ları kaydeder, LLM'e sadece izin verilen tool şemalarını gösterir ve her çağrıda allowlist'i tekrar kontrol eder. |
| `app/tools/calculator.py` | `+ - * /` gibi basit aritmetiği güvenli AST izin listesiyle hesaplar; Python `eval` kullanmaz. |
| `app/tools/csv_analysis.py` | Workspace CSV dosyasındaki tek sayısal kolon için adet, toplam, ortalama, min ve max değerlerini deterministik hesaplar. |
| `app/tools/filesystem.py` | Workspace path sınırı, korunan dizin/dosya kontrolleri ve `file_read`, `file_write`, `directory_list` araçları. Yazma işlemi açık overwrite ve onay kuralları taşır. |
| `app/tools/search.py` | İnterneti değil, workspace içindeki metin dosyalarını arar. Eşleşen dosya/satırları verir. |
| `app/tools/wikipedia.py` | Wikipedia makalesi arar; dil ve alternatif başlık fallback'leriyle kaynaklı içerik döndürür. |
| `app/tools/web_search.py` | Kamu web arama sağlayıcısından temizlenmiş başlık, kısa snippet ve HTTPS URL sonuçları üretir. |
| `app/tools/function_test.py` | Bir saf aritmetik Python fonksiyonu ve JSON test örneklerini okuyup dar kapsamlı test eder. Genel amaçlı Python çalıştırıcısı değildir. |
| `app/tools/python_exec.py` | Kısıtlı Python parçasını child process içinde çalıştıran tool sarmalayıcısı. Kodda bulunması, runtime tarafından kaydedildiği anlamına gelmez; mevcut runtime registry kaydı kontrol edilmelidir. |
| `app/tools/_python_runner.py` | `python_exec` için child process'te çalışan izinli Python altkümesini doğrular ve çıktı sınırlarını uygular. |
| `app/tools/__init__.py` | Tools paketinin işaret dosyası/açıklaması. |

### Bellek, güvenlik, hata ve gözlemleme

| Dosya | Ne yapıyor? |
|---|---|
| `app/memory/store.py` | Kullanıcının açıkça kaydettiği kalıcı bilgileri yerel SQLite'a yazma, listeleme, güncelleme ve silme. |
| `app/memory/context.py` | Saklanan bellekleri LLM çağrısına, ortak backend'i değiştirmeden ekler. Bellek her konuşma mesajı değildir; kullanıcı tarafından kaydedilmiş girdilerdir. |
| `app/memory/__init__.py` | Memory paketinin işaret dosyası/açıklaması. |
| `app/security/approvals.py` | Dosya yazma gibi değişiklik yapan tool'lar için açık onay arayüzü. Onay olmazsa yazma reddedilir. |
| `app/security/__init__.py` | Security paketinin işaret dosyası/açıklaması. |
| `app/errors.py` | İç exception'ları kararlı kullanıcıya dönük hata kodları/metinlerine dönüştürür; ham traceback/model çıktısını göstermemeye çalışır. |
| `app/config/settings.py` | `.env` ve ortam değişkenlerinden ayarları doğrular: model yolu, zaman aşımı, web lookup gibi ayarlar. Gizli değerleri kaynak kontrolüne ekleme. |
| `app/config/logging.py` | Prompt veya credential içeriğini kaydetmeden uygulama loglamasını kurar. |
| `app/config/__init__.py` | Config paketinin işaret dosyası/açıklaması. |
| `app/observability/events.py` | Agent, tool ve görev geçişleri için sınırlı ve içerik güvenli olay izleri üretir. |
| `app/observability/llm.py` | LLM çağrılarının süre/token gibi metadata ölçülerini kaydeder; prompt içeriğini izlere koymayan sarmalayıcıdır. |
| `app/observability/__init__.py` | Observability paketinin işaret dosyası/açıklaması. |

### Kalite ölçümü

| Dosya | Ne yapıyor? |
|---|---|
| `app/evaluation/metrics.py` | Eval case, gözlem, başarı puanı ve toplam ölçüm veri sözleşmelerini/hesaplarını içerir. |
| `app/evaluation/runner.py` | `eval/cases.json` senaryolarını izole workspace'lerde aynı runtime/router yolundan çalıştırır ve raporlar. |
| `app/evaluation/__init__.py` | Evaluation paketinin işaret dosyası/açıklaması. |

## Testler: `tests/`

Test dosyaları çoğunlukla dış ağa/model indirmesine gitmeden sahte LLM cevaplarıyla uygulama akışını sınar. Bir dosya adına tıklayıp ilgili modüldeki örnekleri okuyabilirsin.

| Test dosyası | Kontrol ettiği alan |
|---|---|
| `tests/support.py` | Yazma izni gereken testler için ortak açık-onay fixture'ı. |
| `tests/test_agent.py` | SingleAgent tool kullanımı, araştırma fallback'i, coder akışı ve limitler. |
| `tests/test_api.py` | HTTP task lifecycle, event stream, input/origin kontrolleri, bellek uçları ve konuşma devamı. |
| `tests/test_approval.py` | Dosya yazma onayı, izin reddi ve onay parametrelerinin değiştirilememesi. |
| `tests/test_async_plans.py` | Bağımsız plan adımlarının paralel, bağımlı adımların sıralı çalışması. |
| `tests/test_auto_mode.py` | Sohbet/aritmetik kısa yolları, LLM mode kararı, çok adımlı korumalar ve fallback. |
| `tests/test_coding_demo.py` | Coder'ın örnek Python hatasını düzelterek test etmesi ve Reviewer'ın yeniden kontrolü. |
| `tests/test_config.py` | Model yolu, model biçimi, timeout sınırı ve web lookup ayarı. |
| `tests/test_context.py` | Uzun geçmiş/bağlam sınırları, plan çıktılarının doğru workera verilmesi ve memory bütçesi. |
| `tests/test_conversation.py` | Sohbet geçmişi, takip sorusu, isim hatırlama ve server restart sonrası geçmiş. |
| `tests/test_csv_analysis.py` | CSV hesaplarının doğruluğu ve workspace sınırı. |
| `tests/test_end_to_end.py` | Planner'dan worker, tool, Reviewer ve rapor çıktısına çok aşamalı örnek. |
| `tests/test_error_handling.py` | Timeout, context overflow, geç tool sonucu ve kullanıcıya güvenli hata dönüşü. |
| `tests/test_evaluation.py` | Eval metrikleri, geçerli senaryo tanımı, Türkçe yanıt ve gerekli kanıt kontrolleri. |
| `tests/test_evidence.py` | Wikipedia kanıtının final yanıtta korunması ve kaynak linklerinin eklenmesi. |
| `tests/test_function_test.py` | Dar kapsamlı function_test doğrulaması; import/path engelleri. |
| `tests/test_graph.py` | LangGraph modunun supervisor-worker-reviewer geçişleri, geçmiş ve limitleri. |
| `tests/test_llm_client.py` | Model yanıt normalizasyonu, JSON schema iletimi, bulunmayan model ve hata davranışı. |
| `tests/test_main.py` | CLI davranışı ve beklenmeyen hataların güvenli sunulması. |
| `tests/test_memory.py` | Kalıcı bellek kayıt/silme, secret reddi ve belleğin sonraki LLM çağrısına eklenmesi. |
| `tests/test_model_strategy.py` | Agent rollerinin ortak model instance'ı paylaşması ve rol client seçimi. |
| `tests/test_observability.py` | İzlerde süre/token/kimlik olması ama prompt/tool içeriklerinin sızmaması. |
| `tests/test_planner.py` | Plan ID/bağımlılık doğrulama, hatalı plan retry'si ve planlı supervisor yürütmesi. |
| `tests/test_python_exec.py` | Kısıtlı Python altkümesi, import/dosya engeli, timeout ve çıktı üst sınırı. |
| `tests/test_repeated_tool_errors.py` | Aynı tool hatası tekrarının durdurulması ve kısmi tool kayıtlarının korunması. |
| `tests/test_reviewer.py` | Reviewer kanıt gereksinimi, kaynak incelemesi, kod/rapor kontrolleri ve güvenli abstention. |
| `tests/test_router.py` | Tek worker routing, confidence eşiği, supervisor fallback'i ve sıralı iş koruması. |
| `tests/test_sales_demo.py` | CSV -> Markdown raporu -> deterministik ölçüm kontrolü örneği. |
| `tests/test_search.py` | Workspace metin araması, korunan dosyalar ve symlink sınırı. |
| `tests/test_security.py` | Dosya içindeki prompt injection'ın tool izni/onayı verememesi. |
| `tests/test_state.py` | Görev state geçişi ve nesneler arasında mutable state sızıntısının olmaması. |
| `tests/test_structured.py` | JSON karar tipleri, tool argümanı şeması ve bounded retry. |
| `tests/test_supervisor.py` | Delege etme, uzman seçimi, worker çıktısını birleştirme ve limitler. |
| `tests/test_task_service.py` | Kuyruk, canlı durum, browser approval ve başarısız görevin sonraki tura etkisi. |
| `tests/test_tools.py` | Calculator, dosya araçları, path/symlink kontrolleri ve registry allowlist'i. |
| `tests/test_web_search.py` | Web arama sonucu temizliği, provider hatası, argüman düzeltme ve kaynak bağlantısı. |
| `tests/test_wikipedia.py` | Türkçe/İngilizce Wikipedia fallback, başlık eşleştirme, ayrıştırma ve API hatası. |
| `tests/test_workers.py` | Worker tool izinleri, supervisor delege etmesi ve coder/file_agent ayrımı. |
| `tests/__init__.py` | Test paketinin işaret dosyası. |

## Yapılandırma ve proje dosyaları

| Dosya/klasör | Ne için? |
|---|---|
| `pyproject.toml` | Python paket bağımlılıkları, isteğe bağlı API/graph/test extras ve Ruff/mypy/pytest ayarları. |
| `.env.example` | Yerel ayarlar için örnek anahtar adları. Gerçek `.env` gizlidir ve paylaşılmamalıdır. |
| `.gitignore` | Model, sanal ortam, bellek/veri, secret ve geçici dosyaların Git'e eklenmesini önler. |
| `README.md` | Kurulum, çalıştırma ve proje mimarisi için kullanıcı rehberi. |
| `agent.md` | Agent geliştirme yaklaşımı ve mimari kuralları. |
| `yapilacaklar.md` | Fazlar/roadmap ve proje hedefleri. |
| `SISTEM_TEST_PROMPTLARI.md` | Uygulama üzerinden elle çalıştırılacak test promptları. |
| `CANLI_TEST_RAPORU.md` | Önceki canlı test izlerinin raporu; bugünkü test çalıştırmasının yerine geçmez. |
| `eval/cases.json` | Tekrarlanabilir eval görev tanımları. |
| `examples/` | Satış CSV'si ve küçük Python/test örnekleri. |
| `workspace/` | Agent dosya araçlarının eriştiği çalışma alanı ve örnek senaryolar. |
| `models/` | Yerel model dosyaları; büyük GGUF dosyalarıdır, uygulama kaynak kodu değildir. |
| `.local/` | Yerel SQLite bellek/konuşma gibi makineye özel çalışma verileri. |
| `output/pdf/` | Üretilen PDF belgeleri. |

## Kaynakta dikkat çeken iki ayrıntı

- `app/tools/python_exec.py` ve `app/tools/_python_runner.py` bulunuyor; fakat `app/service/runtime.py` içindeki normal tool registry kurulumunda `PythonExecTool` kaydı yok. Bu nedenle ilgili worker allowlist'ine eklenmiş olması tek başına arayüzden kullanılabildiği anlamına gelmez.
- `app/orchestration/router.py` system promptu researcher için internet yok diyor; web lookup ayarı açıksa `app/agents/workers.py` researcher'a `wikipedia_lookup` ve `web_search` izni verebiliyor. Gerçek çağrı iznini registry + allowlist belirliyor; prompt ifadesi yönlendirmeyi yanıltabilir.

## Nereden başlamalı?

İlk oturumda sadece şu beş dosyayı incelemek tüm resmi kurar: `app/service/runtime.py`, `app/orchestration/auto_mode.py`, `app/agents/supervisor.py`, `app/agents/workers.py` ve `app/agents/single.py`. Sonra `tests/test_auto_mode.py`, `tests/test_supervisor.py`, `tests/test_workers.py` ile anlatılan akışın nasıl sınandığını gör. Dosyaları açmak için editörde `Ctrl+P` yapıp dosya adını yazabilirsin.
