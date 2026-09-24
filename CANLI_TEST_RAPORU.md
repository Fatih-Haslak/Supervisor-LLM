# Yerel multi-agent sistem: canlı test raporu

Tarih: 22 Eylül 2026  
Arayüz: `http://127.0.0.1:8000/`  
Yöntem: Gerçek tarayıcı arayüzünden prompt gönderildi; görev haritası, araç çağrıları ve son yanıt kontrol edildi. Dosya ve CSV sonuçları ayrıca diskten doğrulandı.

| Kapsam | Canlı sonuç |
|---|---|
| Otomatik sohbet, kısa süreli bellek ve sayfa yenileme | `Lale-83` aynı konuşmada hatırlandı; yenilemede geçmiş korundu. |
| Açıkça kaydedilen uzun süreli bellek | CLI ile kaydedilen `Eskişehir` tercihi arayüz yanıtında kullanıldı; test kaydı silindi. |
| `calculator` | `17*23=391`, `3+44=47`, `8*9=72`. |
| `file_read`, `directory_list`, `search` | `MAVI-42` okundu; beş örnek dosya listelendi; `Orion-17` için `brief.txt:2` bulundu. |
| `csv_summary` | Dört satır; toplam 500, ortalama 125, en düşük 80, en yüksek 200. |
| `wikipedia_lookup` | Fatih Tekke sorusunda kaynak bağlantılı yanıt üretildi. Bu araç genel internet araması yapmıyor. |
| `file_write` ve üzerine yazma | Her iki işlemde onay istendi; `overwrite=true` doğru kullanıldı, içerik diskten doğrulandı. |
| Kod agent'ı, `function_test`, reviewer | Hatalı `add` fonksiyonu test kopyasında düzeltildi; 3/3 test ve reviewer geçti. |
| Mesaja yapıştırılan Python kodu | Fibonacci çıktısı 55 ve üstel zaman maliyeti doğru açıklandı; dosya aracı çağrılmadı, reviewer geçti. |
| Çok adımlı plan | Veri agent'ı → rapor yazarı → reviewer; rapor tablo değerleri kaynak CSV ile eşleşti. |
| Router, tek agent, LangGraph, açık supervisor | Her modda seçilen görev ve araç sonucu doğrulandı. Router basit hesap için güven eşiği nedeniyle supervisor'a geçebildi. |
| Hata ve izin sınırları | Olmayan dosya uydurulmadı; `.env` ve `workspace/../` erişimi reddedildi; reddedilen yazma işleminde dosya oluşmadı. |

## Güncelleme: web araştırması ve arayüz belleği (24 Eylül 2026)

| Kapsam | Sonuç |
|---|---|
| Gerçek web sağlayıcısı | Google News RSS üzerinden `Muhammed Salah Premier League takımları performansı kaynaklar` ve `Şenol Güneş` aramaları konuya uygun HTTPS haber kaynakları döndürdü; Bing RSS düşük alaka düzeyi gösterdiği için varsayılan sıradan çıkarıldı. |
| Yerel arayüz | Güncellenmiş arayüz localhost tarayıcısında açıldı; Kalıcı bellek paneli görünür. |
| Kalıcı bellek API akışı | Ayrı geçici SQLite veritabanında kayıt ekleme, listeleme ve silme HTTP üzerinden doğrulandı; deneme kaydı kaldırıldı. |
| Uçtan uca araştırma | Bing ile çalışan ilk Şenol Güneş denemesi researcher → `web_search` → reviewer (`pass`) yolundan tamamlandı. Muhammed Salah denemesi ilgisiz Bing sonuçları ve geçersiz yapılandırılmış model çıktısı sorunlarını ortaya çıkardı; arama sağlayıcısı, kaynak bağlantısı temizliği ve model bozuk çıktısı yedek yanıtı düzeltildi. Yeniden tam model testi uzun üretim nedeniyle henüz doğrulanmadı. Her iki geçici konuşma kaydı da silindi. |
| Otomatik testler | `pytest`: 173 geçti; Ruff temiz; mypy 53 dosyada temiz. Starlette TestClient kaynaklı tek deprecation uyarısı mevcut. |

Web RSS sonuçları başlık ve kısa özetlere dayanır. Kaynak sayfasının tam içeriği
otomatik açılmadığından reviewer özeti ve bağlantıyı denetler; bu, tam metin
doğrulaması anlamına gelmez. Nihai yanıttaki web URL'leri yalnızca arama aracının
döndürdüğü kaynaklarla sınırlandırılır.

## Bulunan ve giderilen sorun

İlk kod düzeltme denemesinde (görev `37eb2e4153a24913918c15eb58749a3c`) planlayıcı inceleme ve düzeltmeyi ayrı adımlara böldü. Reviewer, salt okuma adımında henüz yapılmamış `function_test` sonucunu zorunlu tuttu; görev erken durdu. Reviewer artık geçerli test sonucunu yalnızca Python dosyası gerçekten yazılan agent adımında arıyor. Aynı senaryo yeniden çalıştırıldığında (görev `3f97ae7ecd894b339a2685dd102fc6a9`) 3/3 test ve reviewer ile tamamlandı.

Reviewer gerçekten başarısız olursa durum artık genel `INTERNAL_ERROR` yerine `REVIEW_FAILED` koduyla gösteriliyor. Bu durum için bir regresyon testi eklendi.

## Doğrulama ve son durum

- Planlı rapor görevi `e78741fc3ad740178480014defeda210` başarıyla tamamlandı.
- `.venv\Scripts\python.exe -m pytest -q`: **146 geçti**, bir bağımlılık kaynaklı `DeprecationWarning`.
- `.venv\Scripts\python.exe -m ruff check app/agents/reviewer.py app/errors.py app/service/tasks.py tests/test_reviewer.py tests/test_task_service.py`: **geçti**.
- Test için oluşturulan `ui_` önekli dosyalar kaldırıldı; `workspace/test_scenarios/` içindeki beş örnek dosya korundu.
- Son değişikliklerden sonra yerel sunucu yeniden başlatıldı ve arayüzün açıldığı doğrulandı.

Bu testler denenmiş akışları doğrular. Model yanıtı farklı promptlarda değişebilir; Wikipedia özeti güncel olayların bağımsız doğrulaması değildir.

## 23 Eylül 2026: Triton Server araştırması

Researcher, “Triton Server'ın işlevini araştır” görevinde modelin boş Wikipedia argümanı üretmesi üzerine önce `InvalidArguments` hatasını tekrarlıyordu. Konu çıkarımı Türkçe görev cümleleri için genişletildi. Konu hâlâ belirsizse modelden `title` alanını bir kez düzeltmesi istenir; aynı boş çağrı tekrarlanmaz. Canlı arayüzde görev `f52361e349574e1d8a0a49f258b7b949`, `{"title":"Triton Server"}` argümanıyla `wikipedia_lookup` aracını çağırdı. Türkçe Wikipedia bu konuda makale döndürmedi (`NoArticle`); bu, artık araç girdisi hatası değildir. Sistem kaynak yokluğunu bildirdi. Bilinen `Fatih Tekke` maddesiyle yapılan olumlu denemede Wikipedia aracı ve kaynaklı son yanıt başarılı oldu. Bu konu için genel/üretici belgelerine dayalı araştırma yapmak mevcut Wikipedia aracının kapsamı dışındadır. Son tam paket: **149 test geçti**, Ruff ve mypy kontrolleri geçti.

## 23 Eylül 2026: Muhammed Salah ve kaynak doğruluğu

- Türkçe Wikipedia API'sinde `Muhammed Salah` maddesi var. Kullanıcının “makale yok” yanıtı yanlış bir yokluk hükmüydü. Canlı görev `d1ac130225b646ab9aa94ac5633cf5b2` aynı soruda maddeyi ve bağlantıyı buldu; bu, önceki hatanın her çalışmada oluşmadığını gösterdi.
- `wikipedia_lookup` doğrudan başlıktan sonra başlık araması yapıyor, ayırıcı başlığı (ör. `(futbolcu)`) kontrol ediyor ve uygun Türkçe madde yoksa İngilizce Wikipedia'ya geçiyor. Ad yazılışı varyantları sınırlı ve eşleşen kişinin adı/ayırıcı bilgisi doğrulanıyor. API'nin hız sınırı hatası `NoArticle` yerine `LookupUnavailable` sayılıyor; kısa yeniden deneme ve beş dakikalık sınırlı önbellek eklendi.
- Başarılı Wikipedia araç sonucu supervisor'a kanıt olarak aktarılıyor. Son yanıt “makale yok” derse doğrulanmış başlık ve URL kullanılarak düzeltiliyor; URL unutulursa ekleniyor. İngilizce kaynaktan yapılan model çevirisi farklı bir uyruk uydurabildiği için İngilizce giriş metni özgün haliyle gösteriliyor.
- Canlı arayüz görevi `37bd049559c4421d9412f2e15ef6d54c`, `Muhammed Salah (futbolcu)` isteğine Türkçe madde bağlantısıyla cevap verdi. `acee73742d4a4487932a02d09dccbec2` görevinde 2004 doğumlu başka bir Mo Salah için İngilizce makale seçildi, fakat model `Belgian` yerine yanlışlıkla `Birleşik Krallık` dedi. Düzeltme sonrası `709dd864f958450b9045ed295529f833` görevinde kaynak metni özgün haliyle (`Belgian professional footballer`) ve doğru İngilizce URL ile gösterildi.
- Son tam doğrulama: **158 test geçti**, Ruff ve mypy geçti. Wikipedia'nın makale içeriği ayrıca bağımsız bir ikinci kaynakla doğrulanmıyor; güncel kulüp veya unvan gibi değişken bilgiler bu sınır içinde yorumlanmalı.

## 24 Eylül 2026: Wikipedia araştırmasının Reviewer ile incelenmesi

- Kullanıcının “Şenol Güneş hakkında bilgi getir ve bunu reviewer'a sok” isteği canlı arayüzde önce hatalı biçimde araştırma + rapor yazma planına dönüştü. Onay ekranındaki raporda önceki CSV senaryosunun tutarları da vardı; dosya yazma onayı reddedildi. Bu, reviewer'ın araştırma yanıtı için kaynak kanıtı kabul etmemesi ve açık reviewer isteğinin planlamada dosya üretimi sanılması sorunlarını görünür kıldı.
- Reviewer artık başarılı `wikipedia_lookup` sonucundaki başlık, alıntı metni ve Wikipedia URL'sini kanıt olarak alıyor. Açık reviewer isteğiyle yapılan salt araştırmada plan araştırmacıyla sınırlandırılıyor; araştırma yanıtı reviewer'dan geçiyor, rapor yazarı ve dosya yazma aracı çalışmıyor.
- Canlı arayüzde önce görev `6358b919dc5240fd81d515e1c4fb14c5` ile araştırma + reviewer akışı, sonra ilk kullanıcı ifadesi aynen görev `da6bf8431d0b4d9f8df0360b6abe568e` ile tekrarlandı. İkisinde de Wikipedia aracı başarılı, Reviewer ilk denemede geçti, görev tamamlandı ve dosya yazma/onay isteği oluşmadı.
- Yeni reviewer regresyon testleri Wikipedia kanıtının reviewer'a taşınmasını, açık araştırma-review isteğinde rapor yazma adımının elenmesini ve ilgili CSV rapor akışının bozulmamasını kapsıyor. Son tam paket: **160 test geçti**; Ruff ve mypy geçti. Tek uyarı Starlette test istemcisinin kullandığı AnyIO API'sinin bağımlılık kaynaklı deprecation uyarısı.
