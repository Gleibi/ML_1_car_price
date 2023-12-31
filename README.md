# ML_1_car_price
Создание сервиса предсказания цены автомобиля в рамках ДЗ-1 по машинному обучению

### Что было сделано при подготовке (Части 1-2)
1. Провели разведочный анализ данных, предварительно удалив дубли записи, разделили принзнаки объединенные вместе в рамках одного столбца, привели все к единым единицам измерения внутри одного признака.
2. Заполнили пропуски средним для вещественнозначных признаков и оставили пропуски для категориальных
3. Создали категориальные переменные используя методы OneHot-кодирования
   - При добавлении категориальных фичей я не понял, почему мой датафрейм не сошелся с заданным критерием проверки
4. Построили визуализацию взаимосвязей признаков между собой для зрительной оценки зависимости признаков друг от друга, создали тепловую карту и выделили отдельные пары признаков
    - Видно, что мощность двигателя, сам тип двигателя и год выпуска сильнее всего влияют на цену автомобиля
    - А так же побочные "логичные" связи пробега и года выпуска или двигателя и максимальной мощности

### Обучение моделей (Часть 3)
1. Обучили модель используя метод классической линейной регресии и оценили для нее основные метрики качества для трейна и теста
    - $$R^2_{train} = 0.590, R^2_{test}=0.588$$ 
    - $$MSE_{train} = 117823516473, MSE_{test} = 235115681836$$
    - Подтвердили, что самый большой вес имеет признак максимальная мощность (max_power)
2. Попробовали улучшить предсказания используя Lasso-регрессию
    - $$R^2_{test}=0.561$$ 
    - $$MSE_{test} = 252115742855$$
    - Веса у меньшились, но не изменились, видимо из-за неправильно подбранного гиперпараметра
    - Перебором по сетке (c 10-ю фолдами) пытался подобрать оптимальные параметры для Lasso-регрессии, но удалось только ухудшить результат, хотя и удалось пронаблюдать зануление пары весов
3. Попробовали улучшить предсказания используя Lasso-регрессию
    - $$R^2_{test}=0.561$$ 
    - $$MSE_{test} = 252115742855$$
4. Попробовали улучшить предсказания используя ElasticNet-регрессию
    - $$R^2_{test}=0.561$$ 
    - Результат получился аналогично Lasso регресии, видимо я неправильно работал с данными, хотелось бы на примере увидеть как нужно делать
5. После добавления категориальных фичей и поиска лучшего параметра регуляризации alpha для гребневой (ridge) регрессии удалось достич лучшего результата
    - $$R^2_{test}=0.592$$ 

### Feature Engineering и бизнес (Часть 4)
1. Feature Engineering я не выполнял, пока нет уверенности, что движусь в правильную сторону, хочется получить ОС, прежде чем придумывать новые фичи
2. Оценил  долю предиктов, отличающихся от реальных цен на эти авто не более чем на 10% (в одну или другую сторону) - 22.8%

### Реализация сервиса на FastAPI (Часть 5)
Для простоты решил использовать модель с ElasticNet-регрессией.
Используя проделануую выше работу создал необходимые файлы с параметрами модели и на основе них реализовал:
1. Сервис возвращающий стоимость одной машины, приняв на вход json с её параметрами
2. Севис принимающий на вход csv с признаками множества машин и возврщающий предсказанную стоимость каждой из них

Все созданные файлы разместил в данном git-hub репозитории

#### Пример ответа FastAPI сервиса на запрос в формате JSON
<img width="873" alt="Снимок экрана 2023-11-26 в 23 44 38" src="https://github.com/Gleibi/ML_1_car_price/assets/61700082/9b1a56b6-ec92-46fc-9796-9e96ac7f7c75">

#### Пример ответа FastAPI сервиса на запрос, где на вход получен CSV файл
<img width="905" alt="Снимок экрана 2023-11-27 в 00 50 46" src="https://github.com/Gleibi/ML_1_car_price/assets/61700082/880a7ec1-dc1b-4d14-8ab3-1fbbd2b68c1f">
