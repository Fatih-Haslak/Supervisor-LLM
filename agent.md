# AGENT.md

## Proje amacı

Bu repository, Python ve local LLM kullanarak production mantığına yakın bir **multi-agent / agentic AI sistemi** geliştirmek için oluşturulmuştur.

Ana donanım RTX 4060 Ti 16 GB'dir. Tasarım VRAM verimliliğini gözetmelidir.

## Temel mimari

```text
User
 -> Supervisor
 -> Worker Agent
 -> Tools
 -> Reviewer
 -> Supervisor / Final Answer
```

İlk worker'lar:

- Researcher
- Coder
- FileAgent
- Reviewer

Agent'lar mümkün olduğunca **aynı local LLM backend'ini paylaşmalıdır**. Agent farklılıkları prompt, tool izinleri ve state üzerinden oluşturulmalıdır.

## Geliştirme kuralları

1. `yapilacaklar.md` ana roadmap'tir.
2. Fazları sırayla uygula.
3. Kullanıcı istemedikçe sonraki faza geçme.
4. Her modül küçük, typed ve test edilebilir olsun.
5. Pydantic ile structured output kullan.
6. Tool input'larını mutlaka validate et.
7. Tool'lar `ToolResult` benzeri ortak bir schema döndürsün.
8. Her agent sadece ihtiyacı olan tool'lara erişsin.
9. LLM client ile orchestration katmanını birbirinden ayır.
10. Agent loop'larında hard limit kullan:
   - max steps
   - max tool calls
   - max retries
11. `workspace/` dışına filesystem erişimine izin verme.
12. Secret'ları prompt veya log içine koyma.
13. Shell/Python execution için timeout ve sandbox yaklaşımı kullan.
14. Her önemli değişiklikten sonra testleri çalıştır.
15. Yeni framework eklemeden önce mevcut basit çözümü tercih et.

## Kod kalitesi

- Python 3.11+
- type hints
- async-first API
- Pydantic
- pytest
- ruff
- mypy
- küçük fonksiyonlar
- dependency injection
- açık interface'ler

## Öncelik

Önce çalışan ve anlaşılır sistem:

```text
LLM
-> structured output
-> tools
-> single agent
-> state
-> supervisor
-> sub-agents
-> reviewer
-> memory
-> LangGraph
-> observability
```

Direkt karmaşık multi-agent framework kodu yazma.

## Değişiklik yaparken

Her görevde:

1. İlgili dosyaları incele.
2. Kısa uygulama planı çıkar.
3. Minimum gerekli değişikliği yap.
4. Test ekle veya güncelle.
5. Testleri çalıştır.
6. Hata varsa düzelt.
7. Yapılan değişiklikleri kısa özetle.

Kod çalışmadan görevi tamamlanmış kabul etme.
