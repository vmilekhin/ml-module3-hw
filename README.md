# Module 3 Homework — Python Basics for ML

Домашнее задание по модулю 3: путь пикселя через ResNet18 + основы Python-объектов.

## Задания

1. **`task1_shapes.py`** — shape Tensor после `transform` и `unsqueeze(0)`.
2. **`task2_model_layers.py`** — вывод слоёв ResNet18 (`conv1`, `layer1…layer4`, `avgpool`, `fc`).
3. **`task3_list_copy.py`** — разница между ссылкой на список и копией.
4. **`task4_dataset.py`** — собственный Dataset-класс с `__len__` и `__getitem__`.

## Модульный проект

Код разнесён на модули — как в промышленной разработке:

- **`dataset.py`** — Dataset для 5 пород собак (обёртка над Oxford-IIIT Pet).
- **`model.py`** — ResNet18 с заменённым `fc` (`build_model`).
- **`train.py`** — обучение модели (5 эпох, SGD + momentum).
- **`predict.py`** — инференс: определение породы по фото.

## Результаты

Обучение на 5 породах: **99.62% уверенности** на тестовом фото бигля.

```
epoch=1 avg_loss=0.5942
epoch=2 avg_loss=0.0606
epoch=3 avg_loss=0.0520
epoch=4 avg_loss=0.0301
epoch=5 avg_loss=0.0294
saved: model.pth
Порода: Beagle, уверенность: 99.62%
```

## Запуск

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchvision pillow --index-url https://download.pytorch.org/whl/cpu

# обучение
python train.py

# предсказание
python predict.py путь/к/фото.jpg
```

## Структура

```
ml-module3-hw/
├── dataset.py
├── model.py
├── train.py
├── predict.py
├── task1_shapes.py
├── task2_model_layers.py
├── task3_list_copy.py
├── task4_dataset.py
├── .gitignore
└── README.md
```
