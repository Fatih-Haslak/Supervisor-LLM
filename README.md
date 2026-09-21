# Local LLM Agent System

Bu depo, `yapilacaklar.md` yol haritasının **Faz 0–25** uygulamasıdır.
GGUF modeli Python sürecinde doğrudan yüklenir; LM Studio sunucusu ve API token
gerekmez. Tek agent döngüsü, görev başına merkezi `AgentState`, supervisor
yönlendirmesi ve uzman worker'lar vardır.

## Hangi çalışma biçimini kullanmalıyım?

Arayüzün varsayılanı **Otomatik**. Sıradan mesajlar doğrudan sohbete gider;
hesaplama ve dosya görevlerinde araç kullanan agent'lar devreye girer. Dosya ve
çok adımlı isteklerde supervisor uzman worker'lara görev dağıtır:

```text
Kullanıcı → Supervisor/Planner → uzman agent → araçlar → Reviewer → tek yanıt
                         └──────────── tek yerel GGUF modeli ────────────┘
```

Diğer çalışma biçimleri aynı model, araçlar ve worker'lar üzerinde farklı
yönlendirme yollarını karşılaştırmak içindir. `Supervisor` adımları planlamadan
anlık delege eder; `Router` basit görevleri tek worker'a gönderir; `Tek agent`
uzmanlar arasında delege etmez; `LangGraph` benzer supervisor akışını bir grafik
motorunda yürütür. Arayüzde bunlar gelişmiş seçenekler altında bulunur.

İlk satış demosu `data_agent → writer → reviewer` zinciriyle gerçek yerel modelde
çalıştırıldı. Reviewer rapor sayılarını kaynak CSV'den yeniden hesaplar. Kod
demosu için `function_test`, tek bir saf aritmetik fonksiyonu JSON test vakalarıyla
değerlendirir; genel Python projesi veya `pytest` çalıştırıcısı değildir.

## Kurulum

Python 3.11 veya üzeri gerekir. Bu makinede Python 3.12 sanal ortamı oluşturuldu.
NVIDIA GPU için `llama-cpp-python` CUDA wheel kurulmalıdır. Windows PowerShell:

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu130 llama-cpp-python
uv pip install -e '.[dev,download,graph,api]'
```

Başlangıç modeli, Qwen'in [Qwen3-8B-GGUF](https://huggingface.co/Qwen/Qwen3-8B-GGUF)
deposundaki Q4_K_M dosyasıdır. Yaklaşık 5 GB boyutlu model Türkçe sohbet için
uygundur. Proje kökünde şunu çalıştırın:

```powershell
.venv\Scripts\hf.exe download Qwen/Qwen3-8B-GGUF Qwen3-8B-Q4_K_M.gguf --local-dir models/qwen3-8b
Copy-Item .env.example .env
```

GGUF dosyası `.env` içindeki `AGENT_MODEL_PATH` ile seçilir. Varsayılan yol,
indirilen Qwen dosyasına ayarlıdır. Model dosyaları Git'e eklenmez.
`AGENT_GPU_LAYERS=-1`, desteklenen tüm katmanları GPU'ya gönderir.
CUDA wheel için `cudart64_13.dll` ve `cublas64_13.dll` bulunmalıdır. Windows'ta
DLL'ler PATH üzerinde değilse klasörlerini `AGENT_CUDA_DLL_DIRECTORY` ile verin.
Bu makinede Qwen modeli yüklüyken GPU kullanımı 6.627 MiB ölçüldü; toplam
16.380 MiB VRAM var.

```powershell
.venv\Scripts\python.exe -m app.main --prompt "Merhaba, kısa bir Türkçe cümle yaz."
.venv\Scripts\python.exe -m app.main
```

İkinci komut etkileşimli sohbet açar. `çık`, `exit` veya `quit` ile kapanır.
Bu bilgisayardaki genel `python` Conda'ya aittir ve `llama_cpp` kurulu değildir;
her komutta yukarıdaki `.venv\Scripts\python.exe` yolunu kullanın.

Tool kullanan tek agent için:

```powershell
.venv\Scripts\python.exe -m app.main --agent --show-tools
.venv\Scripts\python.exe -m app.main --agent --show-tools --prompt "25*17 sonucunu hesap makinesiyle bul"
.venv\Scripts\python.exe -m app.main --agent --supervisor --show-tools --prompt "3+44 işlemini hesapla"
.venv\Scripts\python.exe -m app.main --agent --supervisor --plan --show-tools --prompt "workspace dosyalarını listele"
.venv\Scripts\python.exe -m app.main --agent --router --show-tools --prompt "3+44 işlemini hesapla"
.venv\Scripts\python.exe -m app.main --agent --supervisor --show-tools --prompt "workspace içindeki belgelerde Fatih Tekke adını ara"
.venv\Scripts\python.exe -m app.main --agent --graph --show-tools --prompt "3+44 işlemini hesapla"
.venv\Scripts\python.exe -m app.main --agent --graph --trace --show-tools --prompt "3+44 işlemini hesapla"
```

`--show-tools`, çağrılan araçların yalnızca adlarını gösterir. `--allow-python`
eklenirse kısıtlı Python aracı açılır. Bu araç ayrı subprocess, süre sınırı,
çıktı sınırı ve izinli sözdizimi kullanır; genel amaçlı güvenli bir işletim
sistemi sandbox'ı değildir. Varsayılan agent izinlerinde kapalıdır.
Etkileşimli agent modunda bir görev hata verirse hata yazdırılır ve yeni `Görev>`
girdisi beklenir. `--prompt` ile tek görev çalıştırıldığında hata çıkış kodu 1'dir.

`--graph`, LangGraph ile ayrı bir supervisor → worker → reviewer akışı çalıştırır.
Mevcut özel orchestration yolu `--supervisor` ile kullanılmaya devam eder.
Graph aynı GGUF modelini, worker'ları, araç kayıt defterini ve `AgentState`'i
kullanır; coder çıktısı reviewer'dan geçer ve başarısız inceleme en fazla iki
kez düzeltmeye döner. İlk supervisor kararı şemada yalnızca worker delege
edebilir; böylece model worker çalışmadan son cevap üretmez. `--graph`,
`--supervisor`, `--router` ve `--plan` ile birlikte kullanılamaz.

`llama-cpp-python`, GGUF dosyasını Python sürecinde çalıştıran llama.cpp
bağlayıcısıdır. Ayrı bir model sunucusu çalıştırmaz. Transformers ile doğrudan
inference da mümkündür, fakat o yol farklı model ağırlıkları ve Python
bağımlılıkları gerektirir.

## Doğrulama

```powershell
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\ruff.exe check .
.venv\Scripts\mypy.exe app
```

`app.llm.client.LLMClient` soyut arayüzdür. `LlamaCppClient` tek model örneğini
paylaşır ve eşzamanlı istekleri sıralar. Mesaj ve yanıt şemaları Pydantic ile
doğrulanır. Testler modeli yüklemeden arayüzü kontrol eder; CLI gerçek GGUF
dosyasıyla test edilir.

## Structured output ve araçlar

`StructuredDecisionClient`, modelden `final_answer` veya `use_tool` kararını
llama.cpp JSON şemasıyla ister. Bu, modelin `{}` gibi geçersiz kararlar
üretmesini sınırlar. Araç kararı her aracın gerçek argüman şemasıyla kısıtlanır;
böylece yerel model serbest JSON alanlarını sonsuza dek üretmez. Pydantic
geçersiz yanıtı reddeder; en fazla iki kez
yeniden deneme yapılır. `ToolRegistry`, aracı yalnızca çağıranın açık izin
listesindeyse gösterir ve çalıştırır. Mevcut araçlar: `calculator`, `file_read`,
`file_write`, `directory_list`, `csv_summary`, `function_test`, isteğe bağlı
`python_exec`.

Dosya araçları sadece `workspace/` altında çalışır. `file_write` mevcut dosyayı
ancak `overwrite=true` verilirse değiştirir. Araç yolları `note.txt` veya
`workspace/note.txt` olarak verilebilir; ikisi de aynı sandbox dosyasını gösterir.
Tool sonuçları ortak `ToolResult`
şemasına döner. `SingleAgent` model kararını alır, izinli aracı çalıştırır,
sonucu modele geri verir ve son cevabı üretir. Döngü en fazla 10 adım ve
8 araç çağrısı sürer.
Tekrarlanan `PermissionDenied` iki çağrıdan sonra `TOOL_FAILURE` ile durur;
başarısız görevin araç çağrıları API'de korunur.

Her görev `AgentState` içinde benzersiz görev kimliği, mesajlar, mevcut agent,
bekleyen/tamamlanan görevler, worker çıktıları, araç sonuçları, adım sayısı ve
son cevabı tutar. Supervisor yapılandırılmış JSON kararıyla görevi izinli
worker'a verir, çıktısını state'e işler ve son cevabı üretir. En fazla altı tur
çalışır. `general` basit ve karma görevleri; `researcher` yerel belge aramasını;
`coder` workspace içi Python kodunu ve sınırlı fonksiyon testlerini; `file_agent`
dosya yönetimini; `data_agent` CSV analizini; `writer` Markdown raporunu üstlenir.
Hepsi aynı GGUF model örneğini kullanır, fakat prompt ve araç izinleri ayrıdır.
`researcher` yazamaz; `file_agent` kod çalıştıramaz. `python_exec` sadece
`--allow-python` ile `general` ve `coder` rollerine açılır.

`search` aracı yalnızca `workspace/` altındaki UTF-8 metinlerde arama yapar;
internete bağlanmaz. Güncel veya doğrulanamayan bilgi sorularında yerel modelin
yanıtı kaynak doğrulaması sayılmaz.

`--plan` seçeneği supervisor'dan önce 1–4 alt görevli yapılandırılmış bir plan
ister. Plan görevlerinin agent adları, benzersiz kimlikleri ve bağımlılıkları
doğrulanır. Bağımlı görevler önceki worker çıktısını bağlam olarak alır ve
yalnızca kendilerine atanan adımı yürütür. Plan bittikten sonra supervisor tek
son cevap üretir. `--supervisor` tek başına önceki dinamik yönlendirme modunu
kullanır.

`--router`, basit tek adımlı görevlerde worker'ı doğrudan seçer. Karar
`agent` ve 0–1 arasında `confidence` alanlarından oluşur. Güven 0,8'in
altındaysa, karar geçersizse veya istek açıkça "önce ... sonra ..." şeklinde
sıralı iş tarif ediyorsa planlı supervisor çalışır. `--show-tools` rota ve
kullanılan araçları gösterir. Böylece basit hesaplarda supervisor'ın ek model
çağrıları yapılmaz.

Supervisor, Coder ve Writer çıktılarını ayrıca read-only Reviewer'a gönderir. Reviewer
yazılan workspace dosyasını `file_read` ile yeniden okur, Python dosyalarında
sözdizimini kodu çalıştırmadan kontrol eder ve yapılandırılmış `pass/fail`
kararı üretir. `function_test` sonucu varsa testleri yeniden değerlendirir;
`csv_summary` sonucu varsa raporun beş sayısını kaynak CSV'den yeniden hesaplar.
Geçmezse sorunları ilgili worker'a iletir; en fazla iki düzeltme denemesi
yapar. Son deneme de başarısızsa görev tamamlanmış sayılmaz ve bağımlı görevler
çalıştırılmaz. `--show-tools` inceleme sonucunu gösterir. `--router` ile seçilen
Coder görevleri de inceleme için planlı supervisor yoluna gider.

## Bellek (Faz 11)

`AgentState` bir görevin mesajlarını, planını, araç sonuçlarını ve worker
çıktılarını görev boyunca tutar. Görevler arası tercihler ve kararlar SQLite'ta
saklanır. Belleğe yalnızca şu açık komutlar yazar; sohbet geçmişi otomatik
kaydedilmez:

```powershell
.venv\Scripts\python.exe -m app.main --memory-save response_language "Türkçe yanıt ver" --memory-category preference
.venv\Scripts\python.exe -m app.main --memory-save project_backend "GGUF modeli Python sürecinde çalışır" --memory-category decision
.venv\Scripts\python.exe -m app.main --memory-list
.venv\Scripts\python.exe -m app.main --memory-delete response_language
```

Sonraki sohbet ve agent görevleri en güncel 20 kaydı model bağlamında kullanır.
Etkileşimli oturum açıksa yeni kayıtların görünmesi için oturumu yeniden başlatın.
Güncel kullanıcı isteği kayıtla çelişirse güncel istek önceliklidir. Bellek
veritabanı varsayılan olarak `.local/agent_memory.sqlite3` yolundadır ve Git
tarafından yok sayılır; `AGENT_MEMORY_DB_PATH` ile değiştirilebilir. API anahtarı,
parola veya benzeri gizli bilgileri belleğe kaydetmeyin. Komut, yaygın gizli
bilgi etiketlerini reddeder; genel amaçlı gizli bilgi tarayıcısı değildir.

## Gözlemlenebilirlik (Faz 13)

Agent görevlerinde `--trace` kullanıldığında her görev için yapılandırılmış JSON
olayları standart hata akışına yazılır. Olaylar `task_id`, `trace_id`, sıra,
zaman damgası, agent geçişleri, model ve araç süreleri, token sayıları, retry
sayısı ve hata türlerini içerir. Prompt, model cevabı, araç argümanları ve araç
çıktıları iz kayıtlarına alınmaz. İz başına en fazla 200 olay tutulur.

```powershell
.venv\Scripts\python.exe -m app.main --agent --supervisor --trace --prompt "3+44 işlemini hesapla"
```

## Hata yönetimi (Faz 14)

CLI hataları `Hata [KOD]: açıklama` biçiminde gösterir; ham model yanıtı veya
Python traceback'i yazdırmaz. Geçersiz yapılandırılmış yanıtlar sınırlı sayıda
yeniden denenir. `LLM_TIMEOUT`, `CONTEXT_OVERFLOW`, `INVALID_OUTPUT`,
`AGENT_LIMIT`, `SUPERVISOR_LIMIT`, `INVALID_INPUT` ve `STORAGE_FAILURE` ayrı
kodlarla bildirilir. Araçlar başarısız olduğunda `ToolResult` içinde
`error_type` ve temiz bir mesaj döndürür; Python aracı kendi subprocess'ini
süre dolunca sonlandırır.

`AGENT_LLM_TIMEOUT_SECONDS` varsayılan olarak 120'dir. Süre aşılırsa çağrı
bekleyen kullanıcıya hata döner; Python sürecindeki yerel model üretimi güvenli
biçimde zorla durdurulamaz. Arka planda tamamlanana kadar paylaşılan modelin
kilidi korunur; sonraki model çağrıları bu işlemi bekleyebilir. Sürekli model
takılmalarında uygulamayı yeniden başlatın.

## Onay ve güvenlik (Faz 15–16)

`file_write` artık onay gerektirir. Etkileşimli terminalde araç adı, doğrulanmış
yol ve yazılacak içerik gösterilir; yalnızca açık `evet`/`e` yanıtı işlemi
başlatır. Terminal yoksa veya onay reddedilirse görev sırasıyla
`APPROVAL_REQUIRED` ya da `APPROVAL_DENIED` koduyla durur. Çok büyük içerikler
terminalde tam gösterilemediği için onaylanmaz. Salt okunur araçlar bu soruyu
sormaz.

Dosya araçları `workspace/` dışını ve `.env`, `.git`, `.venv` gibi korunan
yolları engeller; Windows büyük/küçük harf varyantları, alternatif akış
adları ve aygıt adları da reddedilir. Dizin listeleme korunan yolları göstermez.
Yerel belgelerden gelen komut benzeri metin veri olarak işaretlenir; araç
izinleri ve yazma onayı model metninden bağımsız uygulanır. Python aracı
izinli sözdizimi ve fonksiyonlarla ayrı süreçte çalışır; genel amaçlı OS
sandbox'ı değildir.

## Test ve değerlendirme (Faz 17–18)

`tests/test_end_to_end.py`, planlanan bir CSV okuma → hesaplama → rapor yazma →
reviewer akışını gerçek dosya ve araçlarla, sahte model yanıtlarıyla doğrular.
Diğer testler geçersiz JSON, eksik dosya, izin, onay, timeout ve retry
sınırlarını kapsar.

Yerel modelin örnek görevlerdeki performansını ölçmek için:

```powershell
.venv\Scripts\python.exe -m app.evaluation.runner
```

Görevler `eval/cases.json` içindedir. Dosya örnekleri her vaka için geçici bir
çalışma alanına yazılır; kayıtlı kişisel bellek değerlendirmeye katılmaz.
Rapor görev başarısı, araç/agent seçimi,
ortalama adım ve süre, araç hata oranı, token sayıları, reviewer geçiş oranı ve
retry oranını verir. Örnek set salt okunur işlemlerden oluşur; değerlendirme
çalıştırıcısı yazma onayı vermez. `reviewer_pass_rate`, hiç inceleme olmayan
sette `null` döner.

## Bağlam yönetimi (Faz 19)

Tam mesajlar ve araç sonuçları `AgentState` içinde tutulur. Modele gönderilen
mesaj geçmişi yaklaşık 9.000 karakterle sınırlanır; ilk istek ve en yeni araç
sonucu korunur. Planlı worker yalnızca kendi alt görevi ve bağımlı olduğu
önceki worker çıktısını alır. Reviewer uzun dosyalarda baş ve son parçayı
inceler; Python sözdizimini dosyanın tamamında kontrol eder. Uzun süreli
bellekten en çok 20 kayıt ve yaklaşık 2.000 karakter eklenir. İlk istek veya
atama bütçeye sığmazsa sessizce kesmek yerine `CONTEXT_OVERFLOW` döner.

## Model stratejisi (Faz 20)

`SharedModelStrategy` chat, supervisor, planner, router, reviewer ve worker
rollerinin hepsine aynı yerel GGUF istemcisini verir. `LlamaCppClient` modeli
ilk istekte bir kez yükler ve çağrıları tek kilitle sıraya alır. Bu, 16 GB VRAM'de
aynı anda birden fazla büyük model tutmaz. Worker fabrikası rol bazlı istemci
seçebilecek arayüze sahiptir; varsayılan çalıştırma yalnızca tek modeli kullanır.
İleride farklı modeller atanırsa yükleme ve VRAM boşaltma politikasının ayrıca
tanımlanması gerekir. Faz 21'deki eşzamanlı yürütme bu fazın kapsamına girmez.

## Eşzamanlı çalışma ve kuyruk (Faz 21–22)

Planlı supervisor, birbirine bağımlı olmayan ardışık `researcher` görevlerini
en fazla iki worker ile eşzamanlı çalıştırır. Sonuçları plan sırasıyla
`AgentState` içine kaydeder. Dosya yazabilen worker'lar sıralı çalışır; tek
yerel modelin çağrıları yine istemci kilidiyle sıraya alınır.

HTTP servisi en fazla sekiz bekleyen görev içeren, bellekte tutulan bir kuyruk
kullanır. Aynı anda bir kullanıcı görevi yürütülür; en fazla 100 görev kaydı
tutulur. Sunucu yeniden başlayınca görev kayıtları kaybolur. Redis, Celery ve
kalıcı dağıtık kuyruk bu yerel sürümde kullanılmaz.

## Yerel API, olay akışı ve arayüz (Faz 23–25)

```powershell
.venv\Scripts\python.exe -m app.api --port 8000
```

Tarayıcıda `http://127.0.0.1:8000/` adresini açın. Varsayılan Otomatik mod
sohbet geçmişini aynı konuşmanın sonraki mesajlarına aktarır. Geçmiş
`.local/conversations.sqlite3` içinde kalır; sayfa yenilenince ve sunucu
yeniden başlayınca aynı sohbet geri yüklenir. “Sohbeti sil ve yeni başlat”
bu konuşmanın geçmişini siler. Ayrı bir yeni sohbette önceki sohbette söylenen
adı hatırlaması beklenmez; kalıcı kullanıcı tercihleri için aşağıdaki açık
`--memory-add` komutları kullanılır.

Arayüz kullanıcı görevini,
agent olaylarını, model süresini/token sayılarını, araç çağrılarını ve
sonuçlarını, inceleme kararlarını ve son yanıtı gösterir. Dosya yazma işlemi
olursa doğrulanmış araç argümanları gösterilir; açık onay veya ret beklenir.
Onay 300 saniye içinde gelmezse işlem reddedilir.

İki örnek giriş:

```text
workspace/sales.csv dosyasındaki amount sütununu analiz et; kısa bulgular ve
tüm metriklerle workspace/sales_report.md raporunu yaz ve doğrula.

workspace/buggy_math.py içindeki add hatasını düzelt;
workspace/buggy_math_tests.json vakalarını function_test ile çalıştır ve doğrula.
```

`examples/` altındaki örnek CSV, Python dosyası ve JSON test vakalarını önce
`workspace/` içine kopyalayın. Arayüzde çalışma biçimini değiştirmeniz gerekmez.

API uç noktaları: `POST /tasks`, `GET /tasks/{task_id}`,
`GET /tasks/{task_id}/events` (SSE) ve
`POST /tasks/{task_id}/approval`; ayrıca `GET` ve `DELETE`
`/conversations/{conversation_id}`. İstek örneği:

```json
{"message":"3+44 işlemini hesapla","mode":"auto"}
```

`mode` için `auto`, `single`, `supervisor`, `plan`, `router` veya `graph`
seçilebilir; varsayılan `auto`dur. İlk istekte `conversation_id` verilmez;
yanıttan gelen UUID sonraki isteklerde kullanılır. API ve UI yalnızca loopback bağlantılarına açıktır;
sunucu `127.0.0.1` adresine bağlanır. Bu sürümde çok kullanıcılı yetkilendirme
yoktur; internete açmayın. SSE, olay meta verilerinin yanında yerel görev
anlık görüntüsünü de gönderir; bu görüntü araç argümanlarını ve sonuçlarını
içerir. API modunda kısıtlı Python aracı kapalıdır.
