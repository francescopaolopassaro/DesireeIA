import json, subprocess, sys
from tokenizers import Tokenizer

exe, strings_path = sys.argv[1], sys.argv[2]
tok_files = sys.argv[3:]

TEXTS = [
    "Hello world",
    "Subject: Server down in production\n\nHi team, the main API has been returning 500 errors.",
    " leading space and  double  spaces ",
    "trailing newline\n",
    "tabs\tand\r\nwindows newlines",
    "Il cliente scrive: \"Non riesco ad accedere all'account – la password non funziona più!\" \U0001F615 — urgente.",
    "Caro supporto, ho ricevuto una fattura errata: €1.234,56 invece di €123,45. Potete correggerla?",
    "Bonjour, je n'arrive pas à me connecter à mon compte depuis ce matin.",
    "Hallo, ich habe meine Rechnung noch nicht erhalten. Bitte um schnelle Antwort.",
    "Hola, el servidor no responde y los clientes se quejan. ¿Pueden ayudarnos?",
    "你好，我无法登录我的账户，请帮忙处理。",
    "こんにちは、サーバーがダウンしています。",
    "مرحبا، لا أستطيع تسجيل الدخول إلى حسابي.",
    "Привет, сервер не отвечает уже двадцать минут.",
    "नमस्ते, मेरा खाता लॉगिन नहीं हो रहा है।",
    "mixed: café naïve façade über Ñandú œuvre",
    "numbers 12345 3.14159 1e-5 0xFF 2026-11-03",
    "code: def f(x): return x**2 + 1  # comment",
    "{\"ticket\": {\"id\": 4821, \"priority\": \"high\"}, \"tags\": [\"payment\", \"refund\"]}",
    "emoji \U0001F600\U0001F680\U0001F44D\U0001F3FD and symbols → ∑ ∞ ©",
    "rare chars ᚠᚢᚦ ཀཁ አለ 한국어 안녕하세요",
    "שלוםה, אין לי גישה לחשבון׳ בבקשה־עזרה.",
    "สวัสดี เซิร์ฟเวอร์ล่ม ราคา ฿100",
    "Հայերեն՝ ինչու՞",
    "नमस्ते। सर्वर बंद है॥",
    "ＡＢＣ　：（全角）！【テスト】・",
    "billing: payments and invoices",
    " billing: payments and invoices",
    "level 0: very unhappy",
    "false: no, the statement does not hold",
    "choice question: Route this email to the right team.",
    "a",
    " ",
    "  ",
    "\n",
    "",
    "don't won't it's we're they've I'm you'll he'd",
    "<mask> inside <mask> text",
    "[MASK] inside [MASK] text",
    "word<mask>word  <mask>  end",
    "<bos>text<eos>",
    "The quarterly report discusses market volatility, bond yields and equity valuations in detail. " * 3,
]
json.dump(TEXTS, open(strings_path, "w"), ensure_ascii=True)

bad_total = 0
for tf in tok_files:
    ref = Tokenizer.from_file(tf)
    out = subprocess.run([exe, tf, strings_path], capture_output=True, text=True)
    if out.returncode != 0:
        print(tf, "EXE FAILED", out.stderr.strip()); bad_total += 1; continue
    lines = out.stdout.split("\n")
    special, rows = lines[0], lines[1:1 + len(TEXTS)]
    bad = 0
    for text, row in zip(TEXTS, rows):
        mine = [int(x) for x in row.split()] if row.strip() else []
        theirs = ref.encode(text, add_special_tokens=False).ids
        if mine != theirs:
            bad += 1
            print("  MISMATCH", repr(text[:60]))
            print("    ref ", theirs[:24])
            print("    mine", mine[:24])
    print(tf.replace("\\", "/").split("laya-")[-1], "|", special, "|", len(TEXTS) - bad, "/", len(TEXTS), "identical")
    bad_total += bad
sys.exit(1 if bad_total else 0)
