# Kratak opis sta se desava u kom fajlu

## Pokretanje
Otprilike je isto za svaki fajl, osim sto se u v1 ne napravi .exe ili Debug :)
```bash
cmake -B ./build   # ili gde zelis da ti se nalazi configuracija projekta
cmake --build ./build
cd build/Debug
./projekat.exe
```
## V1
Ne trazi lokalno biblioteku vec samo pokusava da je skine sa github-a.
U main samo osnovno otvaranje kamere, posto nikad ne dobijemo main.exe.

## V2
Potreban je opencv na racunaru, u system variables dodat kao OpenCV_DIR path\opencv\build kao i u path path\opencv\build\x64\vc16\bin . 

U skladu sa tim promeniti 11. liniju u CMakeLists.txt.

Skinuti shape_predictor_68_face_landmarks.dat i izmeniti liniju 63 main.cpp.

- Radi osnovni face detection. Spor je. 

## V3
Onaj koji si ti pronasao. Ne pokusa da skine sa Github-a jer vec pronadje na racunaru.
Kod je isti kao u V1. 