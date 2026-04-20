import requests
import pandas as pd
from typing import List


def query_ollama(prompt: str, model: str = "qwen2.5:0.5b") -> str:
    """
    Отправляет HTTP-запрос к локальному серверу Ollama и возвращает ответ модели.

    Args:
        prompt (str): Текст запроса к LLM.
        model (str): Название модели в Ollama. По умолчанию 'qwen2.5:0.5b'.

    Returns:
        str: Сгенерированный текстовый ответ от модели или сообщение об ошибке.
    """
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }

    try:
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except requests.exceptions.RequestException as e:
        return f"Ошибка HTTP запроса: {e}"
    except ValueError:
        return "Ошибка парсинга JSON ответа."


def run_inference_experiment(prompts: List[str]) -> pd.DataFrame:
    """
    Прогоняет список запросов через LLM и формирует структурированный отчет.

    Args:
        prompts (List[str]): Список из произвольных текстовых запросов.

    Returns:
        pd.DataFrame: Датафрейм отчета с колонками 'Запрос к LLM' и 'Вывод LLM'.
    """
    results = []
    total = len(prompts)

    for i, prompt in enumerate(prompts, 1):
        print(f"[{i}/{total}] Отправка запроса: {prompt}")
        response_text = query_ollama(prompt)
        results.append({
            "Запрос к LLM": prompt,
            "Вывод LLM": response_text
        })

    return pd.DataFrame(results)


if __name__ == "__main__":
    test_prompts = [
        "Объясни в двух предложениях, что такое микросервисная архитектура.",
        "В чем главные преимущества разбиения крупного графа OpenStreetMap на подрегионы для расчета маршрутов?",
        "Как работает механизм self-attention в архитектуре трансформеров?",
        "Напиши простой пример Kafka Consumer на языке Kotlin.",
        "Для чего применяется шардирование в реляционных базах данных?",
        "Объясни концепцию разрешения коллизий в хеш-таблицах.",
        "Назови основные отличия протокола TCP от UDP.",
        "Какие метрики важны при профилировании параллельного кода на CUDA?",
        "Что такое Docker и какую основную проблему при развертывании он решает?",
        "Кратко объясни алгоритм поиска кратчайшего пути Дейкстры."
    ]

    print("Запуск инференса...")
    report_df = run_inference_experiment(test_prompts)

    csv_filename = "inference_report.csv"
    report_df.to_csv(csv_filename, index=False, encoding='utf-8')

    md_df = report_df.replace(r'\n', '<br>', regex=True)

    md_filename = "inference_report.md"
    md_df.to_markdown(md_filename, index=False)

    print(f"\nОтчет инференса сгенерирован и сохранен в файлы {csv_filename} и {md_filename}.")