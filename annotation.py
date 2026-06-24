import json
import time
from pathlib import Path
import requests

# =========================
# CONFIG
# =========================
INPUT_FILE = "keywords_raw/all_keywords.jsonl"
OUTPUT_DIR = "annotations_v3"
PROCESSED_LOG = "annotations_log_v3.json"

MODEL_NAME = "mistral-small3.2:24b"
OLLAMA_URL = "http://localhost:11434/api/generate"
SLEEP_BETWEEN_CALLS = 0.3

OLLAMA_OPTIONS = {"temperature": 0.1, "num_ctx": 4096}

# =========================
# =========================
# СОГЛАСОВАННЫЕ SUBTYPE (тематически сгруппированы)
# =========================
PERCEPTUAL_SUBTYPES = [
    "person type",
    "person property",
    "clothing item",
    "color",
    "nature and landscape",
    "architecture",
    "spatial",
    "animal",
    "action",
    "item",
    "object property",
    "none",
]

INTERPRETIVE_SUBTYPES = [
    "emotion",
    "mood",
    "social role",
    "time",
    "religion",
    "artistic property",
    "color",
    "activity",
    "clothing property",
    "aesthetic evaluation",
    "status marker",
    "none",
]

# =========================
# HELPERS
# =========================
def normalize_to_list(data):
    if isinstance(data, list):
        return data
    elif isinstance(data, dict):
        if "keyword" in data:
            return [data]
        for v in data.values():
            if isinstance(v, list):
                return v
    return []

def parse_json_response(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.replace("```json", "").replace("```", "")
    
    try:
        return json.loads(text)
    except:
        import re
        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            try:
                return json.loads(match.group())
            except:
                pass
    return {}

def ollama_generate(system_prompt, user_prompt):
    payload = {
        "model": MODEL_NAME,
        "system": system_prompt,
        "prompt": user_prompt,
        "stream": False,
        "format": "json",
        "options": OLLAMA_OPTIONS
    }
    
    for attempt in range(3):
        try:
            resp = requests.post(OLLAMA_URL, json=payload, timeout=600)
            resp.raise_for_status()
            return parse_json_response(resp.json()["response"])
        except Exception as e:
            if attempt < 2:
                time.sleep(4)
            else:
                print(f"  Ошибка: {e}")
                return {}

# =========================
# PROMPT ДЛЯ АННОТАЦИИ ОДНОГО СЛОВА
# =========================
def build_annotation_prompt(keyword, context):
    system = """Ты аннотируешь ключевое слово из описания картины.
Твоя задача — выбрать ТОЛЬКО из предложенных вариантов subtype."""

    perceptual_list = "\n".join([f"  - {s}" for s in PERCEPTUAL_SUBTYPES])
    interpretive_list = "\n".join([f"  - {s}" for s in INTERPRETIVE_SUBTYPES])
    
    user = f"""
Ключевое слово: "{keyword}"
Контекст: "{context}"

Верни JSON с полями:
{{
  "cognitive_type": "perceptual | interpretive",
  "subtype": "выбери из списка ниже",
}}

ДОПУСТИМЫЕ ЗНАЧЕНИЯ ДЛЯ SUBTYPE:

Если cognitive_type = "perceptual", выбери ОДНО из:
{perceptual_list}

Если cognitive_type = "interpretive", выбери ОДНО из:
{interpretive_list}

Правила:
- perceptual: наблюдаемое, конкретные объекты, цвета, части тела, одежда, природа
- interpretive: интерпретация, эмоции, социальные роли, искусство, время

Примеры:
1. keyword="красное", context="красное платье" → {{
  "cognitive_type": "perceptual",
  "subtype": "color",
}}

2. keyword="тревожная", context="тревожная атмосфера" → {{
  "cognitive_type": "interpretive",
  "subtype": "emotion",
}}

3. keyword="Мадонна", context="Мадонна с младенцем" → {{
  "cognitive_type": "interpretive",
  "subtype": "religion",
}}

4. keyword="рука", context="рука поднята вверх" → {{
  "cognitive_type": "perceptual",
  "subtype": "person property",
}}

5. keyword="аристократка", context="аристократка в шляпе" → {{
  "cognitive_type": "interpretive",
  "subtype": "status marker",
}}

Верни ТОЛЬКО JSON, без пояснений.
"""
    return system, user

# =========================
# MAIN
# =========================
def main():
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    
    # Загружаем все записи
    records = []
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            records.append(json.loads(line))
    
    # Загружаем лог обработанного
    processed = {}
    if Path(PROCESSED_LOG).exists():
        with open(PROCESSED_LOG, 'r', encoding='utf-8') as f:
            processed = json.load(f)
    
    total_keywords = sum(len(r.get("keywords", [])) for r in records)
    print(f"Всего записей: {len(records)}")
    print(f"Всего ключевых слов для разметки: {total_keywords}")
    
    processed_count = 0
    keyword_count = 0
    
    for record_idx, record in enumerate(records, 1):
        record_id = record["id"]
        
        if record_id in processed:
            print(f"[{record_idx}/{len(records)}] {record_id} уже обработан")
            continue
        
        keywords = record.get("keywords", [])
        if not keywords:
            continue
        
        print(f"\n[{record_idx}/{len(records)}] {record_id}")
        print(f"  Ключевых слов в записи: {len(keywords)}")
        
        annotated_keywords = []
        
        for kw_idx, kw in enumerate(keywords, 1):
            keyword = kw.get("keyword", "")
            context = kw.get("context", keyword)
            
            print(f"    [{kw_idx}/{len(keywords)}] {keyword}")
            
            # Аннотируем одно слово
            sys_prompt, user_prompt = build_annotation_prompt(keyword, context)
            result = ollama_generate(sys_prompt, user_prompt)
            
            # Добавляем keyword и context к результату
            result["keyword"] = keyword
            result["context"] = context
            
            # Валидация subtype
            valid_subtypes = PERCEPTUAL_SUBTYPES if result.get("cognitive_type") == "perceptual" else INTERPRETIVE_SUBTYPES
            if result.get("subtype") not in valid_subtypes:
                print(f"      ⚠️ Невалидный subtype: {result.get('subtype')}, ставим 'none'")
                result["subtype"] = "none"
            
            annotated_keywords.append(result)
            keyword_count += 1
            time.sleep(SLEEP_BETWEEN_CALLS)
        
        # Сохраняем результат
        output = {
            "image_id": record["image_id"],
            "text_column": record["text_column"],
            "source": record["source"],
            "annotations": annotated_keywords,
            "count": len(annotated_keywords)
        }
        
        out_file = Path(OUTPUT_DIR) / f"{record_id}.json"
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        
        processed[record_id] = {"status": "success", "count": len(annotated_keywords)}
        processed_count += 1
        
        # Сохраняем лог
        with open(PROCESSED_LOG, 'w', encoding='utf-8') as f:
            json.dump(processed, f, ensure_ascii=False, indent=2)
        
        print(f"  Сохранено {len(annotated_keywords)} аннотаций")
    
    print(f"\nГотово!")
    print(f"Обработано записей: {processed_count}")
    print(f"Обработано ключевых слов: {keyword_count}")
    print(f"Результаты в {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()