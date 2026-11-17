import os
import re
import json
import math
import pymysql
import numpy as np
from datetime import datetime
from collections import Counter

# ==============================================================================
# 1. KONFIGURASI BASIS DATA MYSQL LARAGON
# ==============================================================================
LARAGON_MYSQL_CONFIG = {
    "host": "127.0.0.1",
    "port": 3306,
    "user": "root",
    "password": "",  # Default Laragon kosong
    "database": "tomatovision_ai",
    "charset": "utf8mb4"
}

# Kata kunci routing logika percakapan
SCAN_KEYWORDS = {"berapa", "jumlah", "total", "hitung", "petik", "layak", "kamera", "foto", "scan", "deteksi"}
SCAN_EXCLUDE_KEYWORDS = {"pasca", "simpan", "peram", "pupuk", "penyakit", "hama", "cara", "dosis", "akar", "daun", "kenapa", "mengapa", "pantat", "hitam"}
DB_KEYWORDS = {"cek database", "status database", "vgfactor", "cek sql", "laragon"}
SYNC_KEYWORDS = {"sync vektor", "sync vector", "reindex", "update vektor"}
LIST_KEYWORDS = {"apa saja", "daftar", "list", "sebutkan", "macam", "jenis", "katalog"}
TECHNICAL_WORDS = {
    "penyakit", "hama", "pupuk", "nutrisi", "dosis", "daun", "buah", "batang",
    "akar", "bunga", "kutu", "ulat", "busuk", "layu", "keriting", "bercak",
    "budidaya", "panen", "peram", "simpan", "wiwil", "kalsium", "npk", "pantat", "hitam"
}

# Sapaan menggunakan pencocokan kata utuh
GREETING_WORDS = {
    "halo", "hai", "helo", "hello", "hei", "hi", "pagi", "siang",
    "sore", "malam", "assalamualaikum", "p", "ping", "tes", "tomat"
}

# ==============================================================================
# 2. MESIN VEKTOR SEMANTIK (VG-FACTOR ENGINE)
# ==============================================================================
class VectorFactorEngine:
    STOPWORDS = {
        "yang", "di", "ke", "dari", "ini", "itu", "untuk", "pada", "dengan",
        "dan", "atau", "apakah", "bagaimana", "gimana", "kenapa", "mengapa",
        "apa", "saja", "dong", "bro", "ya", "aku", "saya", "kamu", "tolong",
        "bisa", "ada", "kalau", "jika", "nya", "buat", "agar", "supaya",
        "tomat", "buah", "tau", "tahu", "tidak", "nggak", "gak", "deh", "sih"
    }

    @classmethod
    def tokenize(cls, text):
        clean = re.sub(r"[^a-z0-9\s]", " ", text.lower())
        words = [w for w in clean.split() if len(w) > 1 and w not in cls.STOPWORDS]
        features = []
        for w in words:
            # Berikan bobot kata penuh 3x lipat agar kata inti tidak tenggelam oleh trigram
            features.extend([f"w_{w}"] * 3)
            # Ekstraksi tri-gram karakter untuk toleransi salah ketik (typo)
            if len(w) >= 4:
                for i in range(len(w) - 2):
                    features.append(f"g_{w[i:i+3]}")
        return Counter(features)

    @classmethod
    def to_vector_json(cls, text):
        counts = cls.tokenize(text)
        norm = math.sqrt(sum(v * v for v in counts.values()))
        normalized = {k: round(v / norm, 5) for k, v in counts.items()} if norm > 0 else {}
        return json.dumps(normalized)

    @classmethod
    def build_document_vector(cls, topic, category, content, dosage_detail=""):
        # Topik diulang agar lebih dominan, dosage_detail diikutkan ke dalam indeks
        weighted_text = f"{topic} {topic} {category} {content} {dosage_detail}"
        return cls.to_vector_json(weighted_text)

    @staticmethod
    def cosine_similarity(vec_a, vec_b):
        if not vec_a or not vec_b:
            return 0.0
        common = set(vec_a.keys()).intersection(set(vec_b.keys()))
        if not common:
            return 0.0
        a = np.array([vec_a[k] for k in common], dtype=np.float32)
        b = np.array([vec_b[k] for k in common], dtype=np.float32)
        return round(float(np.dot(a, b)), 4)

# ==============================================================================
# 3. MANAJER BASIS DATA MYSQL LARAGON
# ==============================================================================
class LaragonMySQLDatabase:
    def __init__(self, config):
        self.config = config
        # Otomatis mengisi baris baru yang vector_json-nya masih NULL, kosong, atau '{}'
        self.sync_missing_vectors(force_all=False)

    def get_conn(self):
        return pymysql.connect(**self.config, connect_timeout=3)

    def sync_missing_vectors(self, force_all=False):
        """Mengisi otomatis kolom vector_json untuk data di knowledge_vectors."""
        try:
            with self.get_conn() as conn:
                with conn.cursor() as cur:
                    if force_all:
                        cur.execute("SELECT id, topic, category, content, dosage_detail FROM knowledge_vectors")
                    else:
                        cur.execute("""
                            SELECT id, topic, category, content, dosage_detail
                            FROM knowledge_vectors
                            WHERE vector_json IS NULL 
                               OR vector_json = '' 
                               OR vector_json = '{}'
                        """)
                    rows = cur.fetchall()
                    if not rows:
                        return 0

                    updated = 0
                    for row_id, topic, category, content, dosage_detail in rows:
                        vec_json = VectorFactorEngine.build_document_vector(
                            topic or "", category or "", content or "", dosage_detail or ""
                        )
                        cur.execute(
                            "UPDATE knowledge_vectors SET vector_json = %s WHERE id = %s",
                            (vec_json, row_id)
                        )
                        updated += 1
                conn.commit()
                if updated > 0:
                    print(f"[VG-Engine] Berhasil auto-index {updated} baris pengetahuan.")
                return updated
        except Exception as e:
            print(f"[VG-Engine Sync Warning]: {e}")
            return 0

    def get_template(self, key):
        """Menarik template teks secara dinamis dari tabel bot_templates."""
        try:
            with self.get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT template_value FROM bot_templates WHERE template_key = %s", (key,))
                    row = cur.fetchone()
                    if row and row[0]:
                        return row[0]
        except Exception:
            pass

        default_templates = {
            "greeting": "Halo! Aku Dr. Lycopersicon, asisten AI siap mendampingi kebun tomatmu 🍅. Silakan tanyakan masalah pemupukan, penyakit daun, hama, budidaya, atau analisis panen tomat hari ini!",
            "scan_empty": "📊 **Hasil Deteksi Kamera/Foto:**\n\nSaat ini belum ada buah tomat yang terdeteksi di kamera atau foto. Arahkan kamera ke buah tomat atau unggah foto yang jelas ya!",
            "scan_report_header": "🍅 **Hasil Deteksi Kamera ({total} Butir Terdeteksi):**\n🔴 Matang: {matang} butir\n🟡 Setengah Matang: {setengah} butir\n🟢 Muda/Mentah: {mentah} butir\n\n💡 **Rekomendasi Tindakan:**",
            "scan_action_ripe": "• {matang} butir matang: Segera petik hari ini untuk konsumsi segar atau pasar lokal.",
            "scan_action_half": "• {setengah} butir setengah matang: Waktu petik terbaik (fase breaker) untuk pengiriman antarkota.",
            "scan_action_unripe": "• {mentah} butir muda: Biarkan di pohon 4–7 hari lagi.",
            "fallback": "Pertanyaan menarik! Topik tersebut belum terindeks di basis data vektor MySQL Dr. Lycopersicon. Ketik **'daftar penyakit'**, **'daftar hama'**, atau **'daftar nutrisi'** untuk melihat topik yang tersedia."
        }
        return default_templates.get(key, "Halo! Ada yang bisa Dr. Lycopersicon bantu seputar tomatmu?")

    def get_catalog_list(self, category_filter=None):
        """Menarik daftar topik dari knowledge_vectors dengan filter kategori."""
        try:
            with self.get_conn() as conn:
                with conn.cursor() as cur:
                    if category_filter:
                        cur.execute(
                            "SELECT DISTINCT category, topic FROM knowledge_vectors WHERE LOWER(category) LIKE %s ORDER BY topic",
                            (f"%{category_filter}%",)
                        )
                    else:
                        cur.execute("SELECT DISTINCT category, topic FROM knowledge_vectors ORDER BY category, topic")
                    rows = cur.fetchall()

            if not rows:
                return f"Belum ada data pada kategori `{category_filter}` di database." if category_filter else "Basis data pengetahuan tomat di MySQL masih kosong."

            grouped = {}
            for cat, top in rows:
                grouped.setdefault(cat, []).append(top)

            output = ["📚 **Daftar Pengetahuan Agronomi Dr. Lycopersicon:**\n"]
            for cat, topics in grouped.items():
                output.append(f"**🏷️ Kategori {cat.upper()} ({len(topics)} topik):**")
                limit_items = topics if category_filter else topics[:15]
                for t in limit_items:
                    output.append(f"  • {t}")
                if not category_filter and len(topics) > 15:
                    output.append(f"  *...dan {len(topics) - 15} topik lainnya (ketik 'daftar {cat}' untuk filter)*")
                output.append("")

            output.append("💡 *Ketik salah satu nama hama, penyakit, atau nutrisi di atas untuk melihat penanganannya.*")
            return "\n".join(output)
        except Exception as e:
            return f"Gagal membaca katalog database: {e}"

    def vector_search(self, query_text, top_k=3):
        """Pencarian semantik Cosine Similarity ke tabel knowledge_vectors."""
        query_vec = json.loads(VectorFactorEngine.to_vector_json(query_text))
        scored_docs = []
        try:
            with self.get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT topic, category, content, dosage_detail, vector_json FROM knowledge_vectors")
                    rows = cur.fetchall()

            seen_topics = set()
            for topic, category, content, dosage_detail, vec_str in rows:
                if not vec_str or topic in seen_topics:
                    continue
                score = VectorFactorEngine.cosine_similarity(query_vec, json.loads(vec_str))
                if score > 0:
                    seen_topics.add(topic)
                    scored_docs.append({
                        "topic": topic,
                        "category": category,
                        "content": content,
                        "dosage_detail": dosage_detail,
                        "vg_factor": score
                    })

            scored_docs.sort(key=lambda x: x["vg_factor"], reverse=True)
        except Exception as e:
            print(f"[MySQL Search Error]: {e}")

        if not scored_docs:
            return None, 0.0, []

        best_doc = scored_docs[0]
        # Ambil rekomendasi topik lain yang relevansinya cukup dekat
        related = [d["topic"] for d in scored_docs[1:top_k] if d["vg_factor"] >= 0.05]
        return best_doc, best_doc["vg_factor"], related

    def log(self, user_msg, ai_reply, topic, vg_factor, source, scan_ctx):
        """Mencatat interaksi percakapan ke tabel chat_logs."""
        try:
            total = scan_ctx.get("total", 0) if scan_ctx else 0
            with self.get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO chat_logs (user_message, ai_response, matched_topic, vg_factor, engine_source, scan_total, created_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """, (user_msg, ai_reply, topic, float(vg_factor), source, total, datetime.now()))
                conn.commit()
        except Exception:
            pass

    def get_summary(self):
        """Menyajikan status metrik database."""
        try:
            with self.get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT COUNT(*) FROM knowledge_vectors")
                    kb_count = cur.fetchone()[0]
                    cur.execute("SELECT COUNT(*) FROM knowledge_vectors WHERE vector_json IS NOT NULL AND vector_json != '' AND vector_json != '{}'")
                    indexed_count = cur.fetchone()[0]
                    cur.execute("SELECT COUNT(*) FROM bot_templates")
                    tpl_count = cur.fetchone()[0]
                    cur.execute("SELECT COUNT(*), AVG(vg_factor) FROM chat_logs")
                    row = cur.fetchone()
                    total_chat = row[0] or 0
                    avg_vg = round(float(row[1] or 0.0), 3)
            return (
                f"🗄️ **Status Basis Data MySQL Laragon (`tomatovision_ai`):**\n"
                f"- Modul Pengetahuan: **{kb_count} record** (`{indexed_count}` siap terindeks vektor)\n"
                f"- Template Respons Aktif: **{tpl_count} template**\n"
                f"- Total Percakapan Tersimpan: **{total_chat} log**\n"
                f"- Rata-rata Skor Kemiripan (`vg_factor`): **`{avg_vg}`**\n\n"
                f"⚡ *Mode Eksekusi: Pure Edge VG-Vector Engine (Offline-Ready)*"
            )
        except Exception as e:
            return f"Database MySQL Laragon Error: {e}"

sql_db = LaragonMySQLDatabase(LARAGON_MYSQL_CONFIG)

# ==============================================================================
# 4. CONTROLLER PERCAKAPAN
# ==============================================================================
def analyze_scan_report(scan_context):
    if not scan_context or not scan_context.get('has_scan') or scan_context.get('total', 0) == 0:
        return sql_db.get_template("scan_empty")

    total = scan_context.get('total', 0)
    matang = scan_context.get('matang', 0)
    setengah = scan_context.get('setengah', 0)
    mentah = scan_context.get('mentah', 0)

    header = sql_db.get_template("scan_report_header").format(
        total=total, matang=matang, tengah=setengah, setengah=setengah, mentah=mentah
    )
    lines = [header]

    if matang > 0:
        lines.append(sql_db.get_template("scan_action_ripe").format(matang=matang))
    if setengah > 0:
        lines.append(sql_db.get_template("scan_action_half").format(setengah=setengah))
    if mentah > 0:
        lines.append(sql_db.get_template("scan_action_unripe").format(mentah=mentah))

    lines.append("\n💬 *Tips Lanjutan: Tanyakan 'cara peram tomat setengah matang' atau 'suhu simpan tomat merah' untuk panduan pasca panen.*")
    return "\n".join(lines)


def get_ai_response(user_msg, scan_context=None):
    msg_clean = user_msg.lower().strip()
    words_list = re.findall(r"[a-z0-9]+", msg_clean)
    words_set = set(words_list)

    # 0. Rute Sinkronisasi Vektor Otomatis
    if any(k in msg_clean for k in SYNC_KEYWORDS):
        force = "all" in msg_clean or "semua" in msg_clean
        count = sql_db.sync_missing_vectors(force_all=force)
        reply = f"🔄 **Sinkronisasi VG-Factor Selesai:** `{count}` baris pengetahuan berhasil diindeks ke format vektor `w_` + `g_` tri-gram!"
        sql_db.log(user_msg, reply, "vector_sync", 1.0, "VG-Indexer", scan_context)
        return reply

    # 1. Rute Status & Audit Database
    if any(k in msg_clean for k in DB_KEYWORDS):
        reply = sql_db.get_summary()
        sql_db.log(user_msg, reply, "db_status", 1.0, "SQL-Inspector", scan_context)
        return reply

    # 2. Rute Pertanyaan Hasil Kamera YOLOv8
    has_scan_intent = any(k in msg_clean for k in SCAN_KEYWORDS)
    has_agronomy_exclude = bool(words_set & SCAN_EXCLUDE_KEYWORDS)
    if has_scan_intent and not has_agronomy_exclude:
        reply = analyze_scan_report(scan_context)
        sql_db.log(user_msg, reply, "scan_analysis", 0.98, "YOLO-Scan", scan_context)
        return reply

    # 3. Rute Permintaan Daftar / Katalog
    has_list_intent = any(lk in msg_clean for lk in LIST_KEYWORDS)
    cat_map = {
        "hama": "hama",
        "penyakit": "penyakit",
        "jamur": "penyakit",
        "bakteri": "penyakit",
        "pupuk": "nutrisi",
        "nutrisi": "nutrisi",
        "budidaya": "budidaya",
        "panen": "pasca_panen"
    }
    matched_cat = next((v for k, v in cat_map.items() if k in words_set), None)
    if has_list_intent and (matched_cat or "topik" in words_set or "pengetahuan" in words_set):
        reply = sql_db.get_catalog_list(category_filter=matched_cat)
        sql_db.log(user_msg, reply, "catalog_list", 1.0, "SQL-Catalog", scan_context)
        return reply

    # 4. Rute Sapaan Murni
    is_pure_greeting = bool(words_set & GREETING_WORDS) and not bool(words_set & TECHNICAL_WORDS) and len(words_list) <= 3
    if is_pure_greeting:
        reply = sql_db.get_template("greeting")
        sql_db.log(user_msg, reply, "greeting", 1.0, "VG-Conversational", scan_context)
        return reply

    # 5. Pencarian Semantik Menggunakan VG-Factor Engine (Threshold aman: 0.05)
    best_doc, vg_score, related_topics = sql_db.vector_search(user_msg, top_k=3)
    print(f"[VG-Engine] Query: '{user_msg}' | Topik: {best_doc['topic'] if best_doc else 'None'} | VG-Factor: {vg_score}")

    matched_context = best_doc if (best_doc and vg_score >= 0.05) else None

    # 6. Jawaban Murni Berbasis Vektor Semantik MySQL
    if matched_context:
        related_section = ""
        if related_topics:
            links = " • ".join([f"`{t}`" for t in related_topics])
            related_section = f"\n\n🔗 **Topik Terkait:** {links}"

        reply = (
            f"### 🍅 Analisis Agronomi: {matched_context['topic']}\n\n"
            f"{matched_context['content']}\n\n"
            f"💊 **Rekomendasi Dosis & Penanganan:**\n{matched_context['dosage_detail']}"
            f"{related_section}\n\n"
            f"---\n"
            f"🔍 *Kategori: `{matched_context['category']}` • Kemiripan Vektor (`vg_factor`): `{vg_score}`*"
        )
        sql_db.log(user_msg, reply, matched_context["topic"], vg_score, "Pure-VG-Vector", scan_context)
        return reply

    # 7. Fallback jika Pertanyaan di Luar Indeks Pengetahuan
    fallback_reply = sql_db.get_template("fallback")
    sql_db.log(user_msg, fallback_reply, "general", vg_score, "Fallback", scan_context)
    return fallback_reply