# Ambiente di verifica aarch64 (NEON) via QEMU

Questa cartella e' lo strumento ufficiale per verificare per davvero il
codice NEON del motore, non solo scriverlo a occhio: questa macchina di
sviluppo e' x86 e non ha hardware ARM, ma Docker Desktop puo' emulare
aarch64 tramite QEMU (integrato in `docker buildx`), abbastanza fedelmente
da compilare **ed eseguire** i binari, non solo compilarli.

Usata la prima volta il 2026-09-08 per verificare i kernel NEON di
Q4_K/Q6_K (vedi `docs/engine_gap_analysis.md`, sezione "Supporto ARM/NEON").

## Uso rapido

Prerequisito: Docker Desktop avviato (l'icona nella system tray deve essere
verde/stabile — se e' appena stato avviato, attendere che il motore sia
pronto, `docker version` risponde solo a motore su).

```bash
# dalla radice del repo, in Git Bash:
export MSYS_NO_PATHCONV=1   # necessario: senza, Git Bash storpia i path
                             # che iniziano con / dentro gli argomenti Docker

docker run --rm --platform linux/arm64 \
  -v "/c/Sorgenti/Personal/DesireeIA.LocalIA:/src" \
  desireeia-arm64-dev bash /src/.docker-arm64/verify.sh
```

`verify.sh` compila ed esegue `desireeia_selftest` due volte: con
`-march=armv8-a+dotprod` (Apple Silicon, ARM server moderni) e con
`-march=armv8-a` puro (chip piu' vecchi senza `vdotq_s32` hardware, es.
Raspberry Pi 4 / Cortex-A72) — i due percorsi del codice NEON sono diversi
(vedi `dot_i8x16_neon` in `matmul.cpp`) e vanno verificati entrambi.

**Nota sulle prestazioni**: i tempi misurati sotto QEMU NON sono
rappresentativi delle prestazioni reali (ogni istruzione viene tradotta).
Sotto emulazione e' significativa SOLO la correttezza (passed/failed del
selftest), non i numeri di `desireeia_kernel_bench`.

## Se l'immagine `desireeia-arm64-dev` non esiste piu'

L'immagine vive nel registro locale di Docker, non in questa cartella —
va ricostruita una volta se Docker Desktop viene reinstallato o
l'immagine viene rimossa (`docker rmi`):

```bash
export MSYS_NO_PATHCONV=1
docker build --platform linux/arm64 -t desireeia-arm64-dev "C:\Sorgenti\Personal\DesireeIA.LocalIA\.docker-arm64"
```

(Nota: il path del build context va passato in stile Windows con `docker
build`, a differenza di `-v` con `docker run` che vuole lo stile
`/c/...`. Motivo non chiaro — verosimilmente una differenza fra come `docker
run` e il builder di `docker buildx` normalizzano gli argomenti — ma e'
quello che ha funzionato, verificato empiricamente.)

## File

- `Dockerfile` — `arm64v8/gcc:12` + cmake. Minimale apposta: solo quel che
  serve per configurare e compilare il progetto.
- `verify.sh` — compila ed esegue il selftest nelle due varianti dotprod/no-dotprod.
