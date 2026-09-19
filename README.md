# Local LLM Agent System

Bu depo, `yapilacaklar.md` yol haritasının **Faz 0–8** uygulamasıdır.
GGUF modeli Python sürecinde doğrudan yüklenir; LM Studio sunucusu ve API token
gerekmez. Tek agent döngüsü, görev başına merkezi `AgentState`, supervisor
yönlendirmesi ve uzman worker'lar vardır.

## Kurulum

Python 3.11 veya üzeri gerekir. Bu makinede Python 3.12 sanal ortamı oluşturuldu.
NVIDIA GPU için `llama-cpp-python` CUDA wheel kurulmalıdır. Windows PowerShell:

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu130 llama-cpp-python
uv pip install -e '.[dev,download]'
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
.venv\Scripts\python.exe -m app.main --agent --supervisor --show-tools --prompt "workspace içindeki belgelerde Fatih Tekke adını ara"
```

`--show-tools`, çağrılan araçların yalnızca adlarını gösterir. `--allow-python`
eklenirse kısıtlı Python aracı açılır. Bu araç ayrı subprocess, süre sınırı,
çıktı sınırı ve izinli sözdizimi kullanır; genel amaçlı güvenli bir işletim
sistemi sandbox'ı değildir. Varsayılan agent izinlerinde kapalıdır.
Etkileşimli agent modunda bir görev hata verirse hata yazdırılır ve yeni `Görev>`
girdisi beklenir. `--prompt` ile tek görev çalıştırıldığında hata çıkış kodu 1'dir.

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
üretmesini sınırlar. Pydantic geçersiz yanıtı reddeder; en fazla iki kez
yeniden deneme yapılır. `ToolRegistry`, aracı yalnızca çağıranın açık izin
listesindeyse gösterir ve çalıştırır. Mevcut araçlar: `calculator`, `file_read`,
`file_write`, `directory_list`, isteğe bağlı `python_exec`.

Dosya araçları sadece `workspace/` altında çalışır. `file_write` mevcut dosyayı
ancak `overwrite=true` verilirse değiştirir. Araç yolları `note.txt` veya
`workspace/note.txt` olarak verilebilir; ikisi de aynı sandbox dosyasını gösterir.
Tool sonuçları ortak `ToolResult`
şemasına döner. `SingleAgent` model kararını alır, izinli aracı çalıştırır,
sonucu modele geri verir ve son cevabı üretir. Döngü en fazla 10 adım ve
8 araç çağrısı sürer.

Her görev `AgentState` içinde benzersiz görev kimliği, mesajlar, mevcut agent,
bekleyen/tamamlanan görevler, worker çıktıları, araç sonuçları, adım sayısı ve
son cevabı tutar. Supervisor yapılandırılmış JSON kararıyla görevi izinli
worker'a verir, çıktısını state'e işler ve son cevabı üretir. En fazla altı tur
çalışır. `general` basit ve karma görevleri; `researcher` yerel belge aramasını;
`coder` workspace içi Python kodunu; `file_agent` dosya yönetimini üstlenir.
Hepsi aynı GGUF model örneğini kullanır, fakat prompt ve araç izinleri ayrıdır.
`researcher` yazamaz; `file_agent` kod çalıştıramaz. `python_exec` sadece
`--allow-python` ile `general` ve `coder` rollerine açılır.

`search` aracı yalnızca `workspace/` altındaki UTF-8 metinlerde arama yapar;
internete bağlanmaz. Güncel veya doğrulanamayan bilgi sorularında yerel modelin
yanıtı kaynak doğrulaması sayılmaz. Reviewer sonraki fazların konusudur.

`--plan` seçeneği supervisor'dan önce 1–4 alt görevli yapılandırılmış bir plan
ister. Plan görevlerinin agent adları, benzersiz kimlikleri ve bağımlılıkları
doğrulanır. Bağımlı görevler önceki worker çıktısını bağlam olarak alır ve
yalnızca kendilerine atanan adımı yürütür. Plan bittikten sonra supervisor tek
son cevap üretir. `--supervisor` tek başına önceki dinamik yönlendirme modunu
kullanır.
