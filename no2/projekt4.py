import numpy as np
import matplotlib.pyplot as plt


def NPtoquad(RealA22,ImA22,RealA21,ImA21,RealA20,ImA20,RealA2minus1,ImA2minus1,RealA2minus2,ImA2minus2):
    c = 299792458
    G = 6.6743E-11

    o=np.zeros_like(ImA20)

    if (np.array_equal(RealA22, RealA2minus2) and
    np.array_equal(ImA22, -ImA2minus2) and
    np.array_equal(ImA20, o) and
    np.array_equal(RealA21, -RealA2minus1) and
    np.array_equal(ImA21, ImA2minus1)):

        m1 = np.sqrt(32 * np.pi / (15)) * (G / (c ** 4))
        m2 = np.sqrt(16 * np.pi / (5)) * (G / (c ** 4))
        m3 = np.sqrt(4 * np.pi / (5)) * (G / (c ** 4))

        I11 = RealA22 / (2 * m3) - RealA20 / (3 * m1)
        I22 = -RealA22 / (2 * m3) - RealA20 / (3 * m1)
        I33 = 2 * RealA20 / (3 * m1)
        I13 = -RealA21 / m2
        I23 = ImA21 / m2
        I12 = -ImA22 / (2 * m3) #<-|



        return I11, I22, I33, I13, I23, I12

    else:
        raise ValueError("entry data does not meet requirements")

def quadtoNP(I11, I22, I33, I13, I23, I12 ):

    c = 299792458
    G = int(6.6743E-11)

    m1 = np.sqrt(32 * np.pi / (15)) * (G / (c ** 4))
    m2 = np.sqrt(16 * np.pi / (5)) * (G / (c ** 4))
    m3 = np.sqrt(4 * np.pi / (5)) * (G / (c ** 4))

    RealA20 = m1*(I33 - (I11 + I22)/2)
    ImA20 = np.zeros_like(RealA20)
    RealA21 = -m2*I13
    ImA21 = m2*I23
    RealA2minus1 = -RealA21
    ImA2minus1 = ImA21
    RealA22 = m3*(I11 - I22)
    ImA22 = 2*m3*I12
    RealA2minus2 = RealA22
    ImA2minus2 = -ImA22

    return RealA22,ImA22,RealA21,ImA21,RealA20,ImA20,RealA2minus1,ImA2minus1,RealA2minus2,ImA2minus2


def NPsolve (RealA22,ImA22,RealA21,ImA21,RealA20,ImA20,RealA2minus1,ImA2minus1,RealA2minus2,ImA2minus2, theta, phi):

    rely22 = np.sqrt(5 / (64 * np.pi))*np.cos(phi)*(1+np.cos(theta))**2
    imy22 = np.sqrt(5 / (64 * np.pi))*np.sin(phi)*(1+np.cos(theta))**2
    rely21 = np.sqrt(5 / (16 * np.pi))*np.cos(phi)*np.sin(theta)*(1+np.cos(theta))
    imy21 = np.sqrt(5 / (16 * np.pi))*np.sin(phi)*np.sin(theta)*(1+np.cos(theta))
    y20 = np.sqrt(15 / (32 * np.pi))*np.sin(theta)**2
    rely2minus1 = np.sqrt(5 / (16 * np.pi))*np.cos(phi)*np.sin(theta)*(1-np.cos(theta))
    imy2minus1 = -np.sqrt(5 / (16 * np.pi))*np.sin(phi)*np.sin(theta)*(1-np.cos(theta))
    rely2minus2 = np.sqrt(5 / (64 * np.pi))*np.cos(phi)*(1-np.cos(theta))**2
    imy2minus2 = -np.sqrt(5 / (64 * np.pi))*np.sin(phi)*(1-np.cos(theta))**2



    hplus = rely22 * RealA22 + rely21 * RealA21 + y20 * RealA20 + rely2minus1 * RealA2minus1 + rely2minus2 * RealA2minus2 -(imy22 * ImA22 + imy21 * ImA21 + imy2minus1 * ImA2minus1 + imy2minus2 * ImA2minus2)
    hcross = -(rely22 * ImA22 + rely21 * ImA21 + y20 * ImA20 + rely2minus1 * ImA2minus1 + rely2minus2 * ImA2minus2) -(imy22 * RealA22 + imy21 * RealA21 + imy2minus1 * RealA2minus1 + imy2minus2 * RealA2minus2)




    return hplus, hcross

def quadsolve (I11, I22, I33, I13, I23, I12, theta, phi):

    Ithetaphi = (I22 - I11)*np.cos(theta)*np.sin(phi)*np.cos(phi) + I12*np.cos(theta)*(np.cos(phi)**2 - np.sin(phi)**2) + I13*np.sin(theta)*np.sin(phi) - I23*np.sin(theta)*np.cos(phi)
    Iphiphi = I11*np.sin(phi)**2 + I22*np.cos(phi)**2 - 2*I12*np.sin(phi)*np.cos(phi)
    Ithetatheta = (I11*np.cos(phi)**2 + I22*np.sin(phi)**2 + 2*I12*np.sin(phi)*np.cos(phi))*np.cos(theta)**2 + I33*np.sin(theta)**2 - 2*(I13*np.cos(phi) + I23*np.sin(phi))*np.sin(theta)*np.cos(theta)

    hplus = Ithetatheta - Iphiphi
    hcross = 2 * Ithetaphi

    return hplus, hcross



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

hplus1, hcross1 = NPsolve(RealA22, ImA22, RealA21, ImA21, RealA20, ImA20, RealA2minus1, ImA2minus1, RealA2minus2, ImA2minus2, np.pi/2, 0 )

I11, I22, I33, I13, I23, I12 = NPtoquad(RealA22, ImA22, RealA21, ImA21, RealA20, ImA20, RealA2minus1, ImA2minus1, RealA2minus2, ImA2minus2)

hplus2, hcross2 = quadsolve (I11, I22, I33, I13, I23, I12, np.pi/2, 0)

aplus1, bplus1 = np.polyfit(hplus, hplus1, 1)
across1, bcross1 = np.polyfit(hcross, hcross1, 1)

aplus2, bplus2 = np.polyfit(hplus, hplus2, 1)
across2, bcross2 = np.polyfit(hcross, hcross2, 1)

plt.figure()
plt.plot(czas, hplus, marker='.', linestyle='', color="b", markersize=2, label="hplus")
plt.plot(czas, hplus1/aplus1, marker='.', linestyle='', color="r", markersize=2, label=f"hplus1/aplus1, aplus1= {aplus1}" )
plt.title("hplus z harmonijek ")
plt.legend()

plt.figure()
plt.plot(czas, hcross, marker='.', linestyle='', color="b", markersize=2, label="hcross")
plt.plot(czas, hcross1/across1, marker='.', linestyle='', color="r", markersize=2, label=f"hcross1/across1, across1={across1}")
plt.title("hcross z harmonijek ")
plt.legend()

plt.figure()
plt.plot(czas, hplus, marker='.', linestyle='', color="b", markersize=2, label="hplus")
plt.plot(czas, hplus2/aplus2, marker='.', linestyle='', color="r", markersize=2, label=f"hplus2/aplus2, aplus2={aplus2}")
plt.title("hplus z momentu kwadrupolowego z założeniem TT")
plt.legend()

plt.figure()
plt.plot(czas, hcross, marker='.', linestyle='', color="b", markersize=2, label="hcross")
plt.plot(czas, hcross2/across2, marker='.', linestyle='', color="r", markersize=2, label=f"hcross2/across2, across2={across2}")
plt.title("hcross z momentu kwadrupolowego z założeniem TT")
plt.legend()

plt.show()

