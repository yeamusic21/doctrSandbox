import textwrap

def chunk_string(text, n):
    """Splits a string into N substrings without cutting off words."""

    words = text.split()
    chunks = []
    chunk_size = len(words) // n

    for i in range(0, len(words), chunk_size):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)

    # Handle any leftover words
    if len(chunks) > n:
        chunks[-2] += ' ' + chunks[-1]
        chunks.pop()

    return chunks



if __name__=="__main__":
    # text = "This is a long string that we want to split into chunks without cutting off words."
    text = """
    NAMP
    DATE
    CITY
    STATE ZIP
    8-3-89
    Riwsew PAY
    JA FF9S6
    This emple oll haadwtiting 1 - baing callected for 186 in toit ing tomipuker recognitica d hand pruted urumber
    and Jetten. Plen print Abe ollowimg characiere 6 the booone that appenr below.
    0123488789
    0123450789
    0123156789
    0123456789
    0133456787
    0/23 Y5E769
    701
    3752
    80750
    960041
    7
    v01
    3752
    80759
    960841
    E
    32123
    832655
    32
    158
    4594
    32103
    83265F
    82
    7481
    8C559
    4102191
    GT
    214
    7481
    8OS39
    419213
    67
    2 Po4'
    03738
    T29608
    7a
    5716
    61738
    729658
    75
    390
    87/6
    10013
    4
    1234
    44012
    109334
    4D
    675
    4238
    #6002
    Ezalakpdabisiresisemefejeslots
    TPHANHZIIEE 4 Nawfiioniece
    PIBRGXONTWOTEFLUOSFIRYBIA
    EMBOOEIATWGTO6TKPZORTIEE DTA
    Pleees ptidk te lvluwing lexk u the bex bolow:
    We, the People of tha United Staiet, io orde lo lcn & more periect Union, mtablish Justice, inaure domesti
    Trsaqulity, provide kr the cmmon Delense, promole ibe general Yraltare, and securs the Bleasingn ol Liberey t
    ounieliok and onr poaterily, do ocdalu ad establish thia CONSTITUTION for the Unitad Statma ot Amerien.
    we, the PecPte OA fe. Umiteg STetas,11 omerto
    form A More parfect DaLon, establish Jistee,
    IASDr demestic Taneulsy/Omusacer tha.
    don mon Depense pnomo te the oenera L Welfare
    ana Secure The Blessngs of hberty -8 our-
    Belves anol por Pesterity,do o. rdain ang
    establish this RONSTITUTIO N Por the
    Unted d SYates of A merica.
    """
    n = 10
    # print(chunk_string(text, n))
    for i, chunk in enumerate(chunk_string(text, n)):
        print("--------------------CHUNK ", str(i))
        print(chunk)