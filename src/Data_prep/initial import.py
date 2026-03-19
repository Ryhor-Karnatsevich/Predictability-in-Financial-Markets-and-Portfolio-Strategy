import pandas as pd
import glob
import os

# 1. Где лежат исходные файлы
input_path = r"Data_prep\dataset\stocks\*.csv"
output_file = "../../Data/all_stocks.csv"

# Создаем папку Data_prep, если ее нет
if not os.path.exists("../../Data"):
    os.makedirs("../../Data")

# Получаем список всех путей к файлам
all_files = glob.glob(input_path)

print(f"Найдено файлов: {len(all_files)}")

# 2. Создаем (или очищаем) итоговый файл
# Мы запишем в него заголовок первого файла, чтобы создать структуру
first_file = pd.read_csv(all_files[0])
first_file["Ticker"] = os.path.basename(all_files[0]).replace(".csv", "")
first_file.to_csv(output_file, index=False)

print("Объединяю файлы... Погнали!")

# 3. Цикл объединения
# Начинаем со второго файла, так как первый уже записали
for file in all_files[1:]:
    # Читаем текущий файл
    df = pd.read_csv(file)

    # Достаем тикер из названия файла (понятным способом)
    file_name = os.path.basename(file)  # результат: "AAPL.csv"
    ticker = file_name.replace(".csv", "")  # результат: "AAPL"

    # Добавляем колонку
    df["Ticker"] = ticker

    # Дозаписываем в конец итогового файла (mode='a' - append, header=False - без заголовка)
    df.to_csv(output_file, mode='a', index=False, header=False)

    # Чтобы ты видел прогресс и не думал, что всё зависло
    if len(all_files) % 100 == 0:
        print(f"Обработан файл: {ticker}")

print(f"Готово! Все данные собраны в {output_file}")