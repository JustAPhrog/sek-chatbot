# Projekt bota to rozmowy z człowiekiem

Projekt wykonałem przy użyciu pythona i sieci neuronowej. Mój model jest sekwencyjny z dwoma warstwami odrzutu, warstwą wejścia i wyjścia oraz jedną ukrytą.

## Instalacja

Program wymaga zainstalowanego Python 3.9. Lista wymaganych bibliotek jest zapisana w pliku `Pipfile`.

## Pliki programu

Program jest uruchamiany przy pomocy pliku `main.py`. W metodzie `main()` pierwsza instrukcja to pobranie pliku json to nauczenia naszego bota i tu należy podać ścieżkę do niego. Przykładowy plik nazywa się `example_data.json`. Sieć neuronowa jest klasą zdefiniowaną w pliku `chatbot.py`.

## Uruchomienie

Po poprawnie zainstalowanych bibliotekach uruchamiamy skrypt `main.py`. Program w pierwszej kolejności załaduje biblioteki i pobierze potrzebne zależności. W kolejnym kroku stworzy model i wytrenuje go (będzie ukazana informacja na ekranie w postaci informacji o treningu kolejnych epok). Następnie zostanie wyświetlony komunikat, że można zacząć pisać (tylko w języku **angielskim**) i zostanie wyświetlony znak zachęty: `> `. Program jest uruchomiony w nieskończonej pętli, więc należy przerwać jego działanie, aby zakończyć program.