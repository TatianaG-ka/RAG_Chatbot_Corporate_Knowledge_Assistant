# Przewodnik DEMO RAG — „Asystent Wiedzy BRH" (korpus fikcyjny)

> Zestaw **w pełni fikcyjnych** dokumentów banku rozwoju **Bank Rozwoju Horyzont S.A. (BRH)** do zbudowania „Asystenta Wiedzy" na demo.
> Wszystko wytworzone na potrzeby portfolio — **zero ryzyka praw autorskich, znaków towarowych i poufności.** Korpus jest Twój.

---

## Dlaczego korpus fikcyjny, a nie dokumenty prawdziwego banku
- Dokumenty prawdziwej instytucji (nawet „publiczne") są chronione prawem autorskim, a jej nazwa i logo — znakami towarowymi. Budowanie i pokazywanie na nich publicznego produktu rodzi ryzyko.
- Korpus fikcyjny demonstruje **identyczne** umiejętności (chunking, retrieval, cytaty, `refuse-on-no-context`) bez żadnego ryzyka.
- Na rozmowie ten wybór sam w sobie jest atutem: pokazuje, że rozumiesz **zgodność i własność intelektualną** — co jest sednem stanowiska w banku.

## Zasady demo
- **Mały zbiór** (7 plików) → szybciej i trafniej niż 100 dokumentów.
- **Cytuj nazwę pliku / paragraf** w odpowiedzi — to esencja Twojego use case'u.
- Korpus ma **celową lukę** (brak produktów depozytowych) → do pokazania `refuse-on-no-context`.
- **Przećwicz odpowiedzi przed rozmową** — zero niespodzianek z halucynacją na żywo.

---

## Dokumenty w korpusie (wczytaj wszystkie 7)
1. `BRH_Regulamin_Pozyczka_Cyfryzacja_AI.md` — program pożyczkowy (kwoty, oprocentowanie, nabór).
2. `BRH_Warunki_Gwarancja_de_minimis_Rozwoj.md` — gwarancja dla MŚP (60%, okresy, wyłączenia).
3. `BRH_Gwarancja_EkoHoryzont.md` — gwarancja proekologiczna (80%, audyt, premia).
4. `BRH_Procedura_obslugi_wniosku.md` — procedura wewnętrzna (etapy, SLA, role).
5. `BRH_Klauzula_informacyjna_RODO.md` — przetwarzanie danych (cele, okresy, prawa).
6. `BRH_Polityka_odpowiedzialnego_AI.md` — zasady AI (człowiek decyduje, Rada AI, zakaz auto-decyzji).
7. `BRH_Strategia_2030_skrot.md` — misja, filary, oferta (pytania „co bank robi").

---

## Scenariusz demo — 5 pytań + pułapka (przećwicz wcześniej)

| # | Pytanie | Oczekiwana odpowiedź | Źródło (cytat) |
|---|---|---|---|
| 1 | „Jaka jest minimalna i maksymalna kwota Pożyczki na Cyfryzację i AI?" | 500 000 zł – 10 000 000 zł; do 100% kosztów netto | Regulamin PCA, §4 |
| 2 | „Do jakiej części kredytu sięga gwarancja de minimis «Rozwój»?" | do 60%, maks. 3,5 mln zł | Warunki de minimis, §3 |
| 3 | „Czym różni się «EkoHoryzont» od de minimis «Rozwój»?" | Eko: efektywność energetyczna, 80%, 2,5 mln EUR, audyt 30% / Rozwój: ogólna dla MŚP, 60% | EkoHoryzont §5; de minimis §3 |
| 4 | „Ile trwa rozpatrzenie wniosku o pożyczkę?" | do 30 dni roboczych (przy kompletnej dokumentacji) | Procedura, §4 |
| 5 | „Czy BRH pozwala na w pełni automatyczną decyzję kredytową AI?" | Nie — zabronione, decyzję podejmuje człowiek | Polityka AI, §3 |

### Pytanie-pułapka (najważniejszy moment): **refuse-on-no-context**
> **Pytanie:** „Jakie jest oprocentowanie lokaty terminowej w BRH?"
> **Oczekiwana odpowiedź:** *„Nie znajduję podstawy w dokumentach — nie mogę odpowiedzieć."*
> **Dlaczego:** BRH to bank rozwoju; w korpusie **nie ma** produktów depozytowych. System ma powiedzieć „nie wiem", a nie zmyślić.

To różnica między narzędziem dla banku a chatbotem z internetu — **system nie halucynuje.**

### Bonus — poprawna odpowiedź przecząca (kontrast do „refuse")
> **Pytanie:** „Czy gwarancja de minimis «Rozwój» obejmuje kredyt hipoteczny dla osoby fizycznej?"
> **Oczekiwana odpowiedź:** *„Nie — to wyłączenie zapisane w dokumencie"* (cytat: Warunki de minimis, §6).
>
> Pokazuje subtelność: gdy odpowiedź **jest** w dokumencie (nawet przecząca) — system odpowiada z cytatem; gdy jej **nie ma** — odmawia. To dwie różne, pożądane reakcje.

---

## Po demie — jedno zdanie
> *„Zbudowałam to na korpusie fikcyjnego banku rozwoju, który stworzyłam sama — świadomie, żeby nie wykorzystywać cudzych dokumentów bez zgody. Architektura, retrieval, cytaty i odmowa przy braku podstawy działają identycznie na danych wewnętrznych w tenancie — tylko bezpieczniej. To pokazuje, że łączę technologię ze zgodnością."*

---

## Wskazówki techniczne (do Twojego pipeline'u RAG)
- Chunkuj po nagłówkach/paragrafach (§) — struktura dokumentów jest pod to przygotowana.
- W metadanych chunku zapisz `źródło = nazwa pliku` i `paragraf` → łatwy cytat w odpowiedzi.
- Ustaw próg podobieństwa; poniżej progu → `refuse` („nie znajduję podstawy w dokumentach").
- Zachowaj log zapytań i zwróconych źródeł — to Twój „audyt" (argument na rozmowie).
