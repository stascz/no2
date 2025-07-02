import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from scipy.signal import stft
from scipy.fft import fft, fftfreq
from scipy.integrate import simpson

def wczytaj_kolumny_z_pliku(nazwa_pliku):
    kolumny = []

    with open(nazwa_pliku, "r", encoding="utf-8") as plik:
        for linia in plik:
            # Pomija puste linie
            if not linia.strip():
                continue

            # Rozdziel liczby – dowolna ilość spacji jako separator
            wartosci = linia.strip().split()

            # Jeśli to pierwsza linia, inicjalizuj kolumny
            if not kolumny:
                kolumny = [[] for _ in range(len(wartosci))]

            # Dodaj wartości do odpowiednich kolumn
            for i, wartosc in enumerate(wartosci):
                kolumny[i].append(float(wartosc))  # konwertujemy do float

    return kolumny

# Przykład użycia
nazwa_pliku = "dane.txt"
kolumny = wczytaj_kolumny_z_pliku(nazwa_pliku)

czas=np.array(kolumny[0])
hplus=np.array(kolumny[1])
hcross=np.array(kolumny[2])
RealA22=np.array(kolumny[3])
ImA22=np.array(kolumny[4])
RealA21=np.array(kolumny[5])
ImA21=np.array(kolumny[6])
RealA20=np.array(kolumny[7])
ImA20=np.array(kolumny[8])
RealA2minus1=np.array(kolumny[9])
ImA2minus1=np.array(kolumny[10])
RealA2minus2=np.array(kolumny[11])
ImA2minus2=np.array(kolumny[12])
dEdt=np.array(kolumny[13])
Etot=np.array(kolumny[14])


c = 299792458
G = 6.67430e-11
podz1 = 1e32
podz2 = 1e38

A22 = RealA22 + 1j*ImA22
A21 = RealA21 + 1j*ImA21
A20 = RealA20 + 1j*ImA20
A2minus1 = RealA2minus1 + 1j*ImA2minus1
A2minus2 = RealA2minus2 + 1j*ImA2minus2

dA22_dt = np.diff(A22, czas)
dA21_dt = np.diff(A21, czas)
dA20_dt = np.diff(A20, czas)
dA2minus1_dt = np.diff(A2minus1, czas)
dA2minus2_dt = np.diff(A2minus2, czas)

#c**3/(32*np.pi*G)
dE_dt = c**3/(32*np.pi*G)*(abs(dA22_dt)**2 + abs(dA21_dt)**2 + abs(dA20_dt)**2 + abs(dA2minus1_dt)**2 + abs(dA2minus2_dt)**2)

ft2A22_dt = fft(dA22_dt)
ft2A21_dt = fft(dA21_dt)
ft2A20_dt = fft(dA20_dt)
ft2A2minus1_dt = fft(dA2minus1_dt)
ft2A2minus2_dt = fft(dA2minus2_dt)

Fs = 5000

freqs = fftfreq(len(czas), 1/Fs)



inta22 = np.zeros_like(czas, dtype=float)
inta21 = np.zeros_like(czas, dtype=float)
inta20 = np.zeros_like(czas, dtype=float)
inta2minus1 = np.zeros_like(czas, dtype=float)
inta2minus2 = np.zeros_like(czas, dtype=float)

inta22[1] = abs(ft2A22_dt[1])**2
inta21[1] = abs(ft2A21_dt[1])**2
inta20[1] = abs(ft2A20_dt[1])**2
inta2minus1[1] = abs(ft2A2minus1_dt[1])**2
inta2minus2[1] = abs(ft2A2minus2_dt[1])**2


for i in range(len(czas)):


    if i < 2:
        continue

    inta22[i] = simpson(np.abs(ft2A22_dt[:i])**2, freqs[:i])
    inta21[i] = simpson(np.abs(ft2A21_dt[:i])**2, freqs[:i])
    inta20[i] = simpson(np.abs(ft2A20_dt[:i])**2, freqs[:i])
    inta2minus1[i] = simpson(np.abs(ft2A2minus1_dt[:i])**2, freqs[:i])
    inta2minus2[i] = simpson(np.abs(ft2A2minus2_dt[:i])**2, freqs[:i])

#(c**3/(16*np.pi*G))

E_tot = (c**3/(16*np.pi*G))*(inta22 + inta21 + inta20 + inta2minus1 + inta2minus2)

a1, b1 = np.polyfit(dEdt, dE_dt , 1)
a2, b2 = np.polyfit(Etot, E_tot , 1)
print(a1)
print(a2)

plt.figure()
plt.plot(czas, dEdt, marker='.', linestyle='', color="b", markersize=2, label="dane")
plt.plot(czas, dE_dt/a1, marker='.', linestyle='', color="r", markersize=2, label="szukane")
plt.yscale("log")
plt.xlabel("czas po odbiciu (ms)")
plt.ylabel("jasność fali graw. (10^42 erg/s)")
plt.legend()

plt.figure()
plt.plot(czas,Etot , marker='.', linestyle='', color="b", markersize=2, label="dane")
plt.plot(czas,E_tot/a2 , marker='.', linestyle='', color="r", markersize=2, label="szukane")
plt.yscale("log")
plt.xlabel("czas po odbiciu (ms)")
plt.ylabel("energia fali graw. (10^45 erg)")
plt.legend()

plt.show()

