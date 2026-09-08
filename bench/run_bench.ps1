# Benchmark end-to-end SEMPRE con motore aggiornato.
#
# Perche' esiste: il progetto .NET tiene una COPIA PROPRIA di
# DesireeIALocaleEngine.dll nella propria cartella di output, e quella copia
# viene aggiornata solo da `dotnet build` del progetto CLI. Ricostruire solo
# il nativo con cmake NON la aggiorna, quindi la CLI continua a caricare la
# DLL di una build precedente in silenzio.
#
# Costo dell'errore (2026-09-07): un'intera serie di misure end-to-end fatta
# contro un motore vecchio. Le differenze osservate erano deriva termica e
# processi appesi, non le modifiche appena scritte, e su quella base sono
# state prese due decisioni sbagliate (rimozione del path a 4 righe e
# accorciamento dello spin) che sono poi state rimisurate da capo.
#
# Uso:  .\bench\run_bench.ps1 -Model <path.gguf> [-Threads 16] [-Tokens 16]

param(
    [Parameter(Mandatory = $true)][string]$Model,
    [int[]]$Threads = @(8, 16, 20),
    [int]$Tokens = 16,
    [string]$Prompt = "The capital of France is",
    [switch]$Profile,
    # Secondi di riposo prima di OGNI misura. Vedi la nota sul throttling
    # qui sotto: senza pausa si misura la temperatura del portatile, non il
    # codice.
    [int]$RestSeconds = 90
)

# ATTENZIONE ALLA METODOLOGIA — throttling termico su questo portatile.
#
# Misurato il 2026-09-08: TRE esecuzioni consecutive dello stesso identico
# comando, stesso binario, hanno dato 17,57 -> 4,59 -> 3,88 tok/s. Il crollo
# arriva in meno di un minuto di carico AVX2 su 16 thread.
#
# Conseguenza: il "best-of-N" NON va bene qui. Funziona quando il rumore e'
# casuale, ma se la macchina degrada in modo monotono l'unica misura valida
# e' LA PRIMA dopo un periodo di riposo; tutte le successive misurano quanto
# si e' scaldata la CPU. Per questo si riposa prima di ogni configurazione e
# si riporta la prima lettura.
#
# Perche' e' costato caro saperlo: una serie di misure degradate e' stata
# scambiata per una regressione del codice, e sono state avviate indagini su
# modifiche che non c'entravano nulla.
#
# Per misure piu' pulite: chiudere Gestione attivita' (i grafici in tempo
# reale consumano ~19% di un core) e le finestre di Edge/WebView (~14%), e
# impostare il piano energetico su "Prestazioni elevate".

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot

# 1. Processi appesi: rubano core e falsano tutto. Gia' successo con due
#    `find` ricorsivi che avevano accumulato 10,8 e 4,8 ore di CPU.
$hogs = Get-Process -ErrorAction SilentlyContinue |
        Where-Object { $_.CPU -gt 600 -and $_.Name -notmatch 'System|Idle|dwm|explorer|claude|Code' }
if ($hogs) {
    Write-Host "ATTENZIONE: processi con molta CPU accumulata (possono falsare la misura):" -ForegroundColor Yellow
    $hogs | Select-Object Name, Id, CPU | Format-Table -AutoSize | Out-String | Write-Host
}

# 2. Nativo, poi CLI. L'ordine conta: e' il build della CLI che copia la DLL.
Write-Host "build nativo..." -ForegroundColor Cyan
cmake --build "$root\native\out" -j 8 2>&1 | Select-String -Pattern "error|Error" | ForEach-Object { Write-Host $_ -ForegroundColor Red }

Write-Host "build CLI (copia la DLL aggiornata)..." -ForegroundColor Cyan
dotnet build "$root\cli\DesireeIA.Cli" -c Release -v q --nologo | Out-Null

$cli = "$root\cli\DesireeIA.Cli\bin\Release\net10.0\desireeia-cli.exe"

# 3. Prova che la copia sia davvero allineata, invece di sperarlo.
$src = Get-Item "$root\native\out\DesireeIALocaleEngine.dll"
$dst = Get-Item "$root\cli\DesireeIA.Cli\bin\Release\net10.0\DesireeIALocaleEngine.dll"
if ($src.Length -ne $dst.Length -or $src.LastWriteTime -ne $dst.LastWriteTime) {
    throw "la DLL della CLI NON e' allineata a native\out: la misura sarebbe contro un motore vecchio."
}
Write-Host ("DLL allineata ({0}, {1} byte)" -f $dst.LastWriteTime, $dst.Length) -ForegroundColor Green

# 4. Correttezza prima della velocita': un motore veloce che sbaglia non serve.
$out = & $cli generate $Model "The capital of France is" --max-tokens 8 2>&1 | Out-String
if ($out -notmatch "Paris") {
    throw "CONTROLLO DI CORRETTEZZA FALLITO: atteso 'Paris' nell'output.`n$out"
}
Write-Host "correttezza: ok (Paris)" -ForegroundColor Green

foreach ($t in $Threads) {
    if ($RestSeconds -gt 0) {
        Write-Host ("riposo {0}s prima della misura (throttling: vedi nota in testa)..." -f $RestSeconds) -ForegroundColor DarkGray
        Start-Sleep -Seconds $RestSeconds
    }
    $sel = if ($Profile) { "decode:|dispatch|seriale:" } else { "decode:" }
    $r = & $cli bench $Model --tokens $Tokens --warmup 2 --threads $t --prompt $Prompt 2>&1 | Select-String $sel
    Write-Host "threads=$t  (prima misura dopo riposo = l'unica valida)"
    $r | ForEach-Object { Write-Host "  $_" }
}
