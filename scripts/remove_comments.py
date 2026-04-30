import argparse
import io
import sys
import tokenize
from pathlib import Path

# Force UTF-8 encoding for standard output
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

def remove_comments_and_docstrings(source_code: str) -> str:
    """
    Видаляє всі коментарі та docstrings з Python коду, зберігаючи його структуру.
    """
    io_obj = io.StringIO(source_code)
    out_tokens = []
    prev_toktype = tokenize.NEWLINE  # Починаємо так, ніби це новий рядок
    last_lineno = -1
    last_col = 0
    nesting_level = 0
    
    try:
        for tok in tokenize.generate_tokens(io_obj.readline):
            token_type = tok.type
            token_string = tok.string
            start_line, start_col = tok.start
            end_line, end_col = tok.end
            
            # Рахуємо рівень вкладеності дужок, щоб не видаляти ключі словників тощо
            if token_type == tokenize.OP:
                if token_string in "([{":
                    nesting_level += 1
                elif token_string in ")]}":
                    nesting_level -= 1
            
            # Відновлення оригінальних відступів
            if start_line > last_lineno:
                last_col = 0
            if start_col > last_col:
                out_tokens.append(" " * (start_col - last_col))
                
            # Пропускаємо однорядкові коментарі
            if token_type == tokenize.COMMENT:
                pass
            # Пропускаємо docstrings (лише поза дужками/списками/словниками)
            elif (token_type == tokenize.STRING and 
                  nesting_level == 0 and 
                  prev_toktype in (tokenize.INDENT, tokenize.NEWLINE, tokenize.NL, tokenize.ENCODING, tokenize.DEDENT)):
                pass
            else:
                out_tokens.append(token_string)
                
            # Оновлюємо попередній токен (ігноруємо кодування та коментарі, щоб не збивати логіку)
            if token_type not in (tokenize.ENCODING, tokenize.COMMENT):
                prev_toktype = token_type
                
            last_col = end_col
            last_lineno = end_line
            
        clean_code = "".join(out_tokens)
    except tokenize.TokenError:
        # У разі синтаксичних помилок повертаємо оригінальний код
        return source_code

    # Видаляємо зайві порожні рядки
    cleaned_lines = [line for line in clean_code.splitlines() if line.strip()]
    return "\n".join(cleaned_lines) + "\n"

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Скрипт для акуратного видалення всіх коментарів та docstrings з Python файлів."
    )
    parser.add_argument(
        "input_file", 
        type=Path, 
        help="Шлях до файлу, який потрібно очистити"
    )
    parser.add_argument(
        "-o", "--output", 
        type=Path, 
        help="Шлях до вихідного файлу (за замовчуванням перезаписує вхідний)"
    )
    
    args = parser.parse_args()
    input_path: Path = args.input_file
    output_path: Path = args.output if args.output else input_path
    
    if not input_path.exists():
        print(f"❌ Помилка: Файл '{input_path}' не знайдено.")
        return
        
    if input_path.suffix != '.py':
        print(f"⚠️ Попередження: Файл '{input_path}' можливо не є Python скриптом.")
        
    try:
        source = input_path.read_text(encoding="utf-8")
        clean_source = remove_comments_and_docstrings(source)
        
        # Створюємо директорії, якщо їх не існує
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(clean_source, encoding="utf-8")
        
        print(f"✅ Успішно! Коментарі видалено. Результат збережено у: {output_path}")
    except Exception as e:
        print(f"❌ Сталася помилка під час обробки файлу: {e}")

if __name__ == "__main__":
    main()
