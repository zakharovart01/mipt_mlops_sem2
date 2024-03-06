# mipt_mlops_sem2
## Описание проекта
Предисловие: суть данного курса заключается в освоение mlops, кажется, что лучше это делать на задаче, в которой результат не сильно важен (то есть возможно я не смогу показать хороший результат на задаче, но постараюсь).
В качестве задачи я выбрал задачу предсказания лэйблов новостей - фейк/не фейк - увы но на англ, т.к. подходящих датасетов на русском не так легко найти - классификаю цифр на картинке, задача является базовой, располагалась на kaggle (данные можно взять от туда), так же её можно решать с помощью свёрпточных нейронных сетей на pytorch.

## Данные
Загуглив на предложенном сайте https://datasetsearch.research.google.com/search?src=3&query=fake%20news&docid=L2cvMTFyeWhqMmpqZw%3D%3D, больше всего мне понравился датасет https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification/data, но можно использовать и другие, возможно удастся найти более знаменитый датасет, чтобы в конце сравнить себя уже существующими решениями.
Если коротко то датасет имеет структуру:
Заголовок - Статья - Тарегет
Возможно данных много, поэтому порежу чуть-чуть

## Подоход к моделированию
Решение будет выполняться с помощью трансформера, библиотека transformers + pytorch, возможно, если будет возможность по времени напишу под pytorch lightning. Краткая схема предсказаний:
Предобработка текста -> токинезация -> подача на вход трансформеру (надеюсь впихнуть bert tiny в 12 гб 3060, должно получиться) -> получения представления токена CLS -> линейный слой/парцептрон для предикта класса
Возможно веса модели слишком большие, если это так, и нормально сдать проект не получится, то сделаю следующим образом:
Предобработка текста -> токинезация -> подача на вход RNN -> получения представления скрытого слоя -> линейный слой/парцептрон для предикта класса

## Продакшн пайплайн
Раз в сутки или в онлайн формате (ну это сильно будет) новость которая приходит будет размечаться, модель будет лежать в виде весов, которые будут обновляться в ручном формате.
Необходимые шаги для Production Pipeline (CI): Data extraction -> Data validation -> Data preparation -> Model training -> Model evaluation -> Model validation
