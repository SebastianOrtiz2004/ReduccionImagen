import math
import random
import numpy as np

MASCARA_GAUSSIANA_3X3 = [
    [1, 2, 1],
    [2, 8, 2],
    [1, 2, 1]
]

MASCARA_QUEMADA_3X3 = [
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
]


def formato_uint8(valor):
    numero = int(valor + 0.5)
    if numero < 0:
        return 0
    if numero > 255:
        return 255
    return numero


def crear_matriz(filas, columnas):
    matriz = []
    for _ in range(filas):
        matriz.append(bytearray(columnas))
    return matriz


def agregar_padding(matriz, filas, columnas, borde_f, borde_c):
    nuevas_filas = filas + (2 * borde_f)
    nuevas_columnas = columnas + (2 * borde_c)
    matriz_pad = crear_matriz(nuevas_filas, nuevas_columnas)

    for f in range(filas):
        for c in range(columnas):
            matriz_pad[f + borde_f][c + borde_c] = matriz[f][c]

    return matriz_pad


def ordenar_elementos(lista):
    datos = list(lista)
    n = len(datos)
    for i in range(n):
        for j in range(0, n - i - 1):
            if datos[j] > datos[j + 1]:
                datos[j], datos[j + 1] = datos[j + 1], datos[j]
    return datos


def obtener_moda(lista, valor_central=None):
    if valor_central is None:
        valor_central = lista[len(lista) // 2]

    n = len(lista)
    max_repeticiones = 0
    for i in range(n):
        actual = lista[i]
        conteo = 0
        for j in range(n):
            if lista[j] == actual:
                conteo += 1
        if conteo > max_repeticiones:
            max_repeticiones = conteo

    empatados = []
    for i in range(n):
        actual = lista[i]
        conteo = 0
        for j in range(n):
            if lista[j] == actual:
                conteo += 1
        if conteo == max_repeticiones and actual not in empatados:
            empatados.append(actual)

    if len(empatados) == 1:
        return empatados[0]

    if valor_central in empatados:
        return valor_central

    mejor_val = empatados[0]
    menor_dist = abs(mejor_val - valor_central)
    for val in empatados:
        dist = abs(val - valor_central)
        if dist < menor_dist:
            menor_dist = dist
            mejor_val = val

    return mejor_val


def procesar_canal(matriz, filas, columnas, mascara, tam_f, tam_c, tipo_filtro, salto_f, salto_c, usar_pad):
    borde_f = (tam_f - 1) // 2
    borde_c = (tam_c - 1) // 2

    if usar_pad:
        matriz_trabajo = agregar_padding(matriz, filas, columnas, borde_f, borde_c)
        limite_filas = filas
        limite_cols = columnas
    else:
        matriz_trabajo = matriz
        limite_filas = filas - (2 * borde_f)
        limite_cols = columnas - (2 * borde_c)

    filas_salida = len(range(0, limite_filas, salto_f))
    cols_salida = len(range(0, limite_cols, salto_c))
    matriz_salida = crear_matriz(filas_salida, cols_salida)

    suma_pesos = 0.0
    es_suma_cero = False
    if tipo_filtro == "Máscara (Convolución)":
        for i in range(tam_f):
            for j in range(tam_c):
                suma_pesos += mascara[i][j]
        if abs(suma_pesos) < 1e-6:
            suma_pesos = 1.0
            es_suma_cero = True

    f_out = 0
    for f in range(0, limite_filas, salto_f):
        c_out = 0
        for c in range(0, limite_cols, salto_c):

            if tipo_filtro == "Máscara (Convolución)":
                acumulador = 0.0
                for i in range(tam_f):
                    for j in range(tam_c):
                        pixel = matriz_trabajo[f + i][c + j]
                        peso = mascara[i][j]
                        acumulador += pixel * peso
                if es_suma_cero:
                    nuevo_valor = abs(acumulador)
                else:
                    nuevo_valor = acumulador / suma_pesos
                matriz_salida[f_out][c_out] = formato_uint8(nuevo_valor)

            elif tipo_filtro == "Mediana":
                vecinos = []
                for i in range(tam_f):
                    for j in range(tam_c):
                        vecinos.append(matriz_trabajo[f + i][c + j])
                ordenados = ordenar_elementos(vecinos)
                matriz_salida[f_out][c_out] = ordenados[len(ordenados) // 2]

            elif tipo_filtro == "Moda":
                vecinos = []
                for i in range(tam_f):
                    for j in range(tam_c):
                        vecinos.append(matriz_trabajo[f + i][c + j])
                pixel_central = matriz_trabajo[f + borde_f][c + borde_c]
                matriz_salida[f_out][c_out] = obtener_moda(vecinos, pixel_central)

            c_out += 1
        f_out += 1

    return matriz_salida, filas_salida, cols_salida


def calcular_matriz_gaussiana(tam):
    sigma = tam / 4.0
    centro = (tam - 1) // 2

    pesos = []
    suma_total = 0.0
    for i in range(tam):
        fila = []
        for j in range(tam):
            x = j - centro
            y = i - centro
            exponente = -((x * x) + (y * y)) / (2.0 * sigma * sigma)
            val = math.exp(exponente)
            fila.append(val)
            suma_total += val
        pesos.append(fila)

    matriz_norm = []
    for i in range(tam):
        fila = []
        for j in range(tam):
            fila.append(pesos[i][j] / suma_total)
        matriz_norm.append(fila)

    return matriz_norm


def aplicar_ruido_sal_pimienta(canal_r, canal_g, canal_b, filas, columnas, densidad=0.05):
    try:
        densidad = float(densidad)
    except (ValueError, TypeError):
        densidad = 0.0

    densidad = max(0.0, min(1.0, densidad))

    nuevo_r = [bytearray(fila) for fila in canal_r]
    nuevo_g = [bytearray(fila) for fila in canal_g]
    nuevo_b = [bytearray(fila) for fila in canal_b]

    if densidad <= 0.0:
        return nuevo_r, nuevo_g, nuevo_b

    total_pixeles = filas * columnas
    num_afectados = int(round(total_pixeles * densidad))
    if num_afectados <= 0:
        return nuevo_r, nuevo_g, nuevo_b

    if densidad <= 0.5:
        indices = random.sample(range(total_pixeles), num_afectados)
        num_pimienta = num_afectados // 2

        for idx in indices[:num_pimienta]:
            f = idx // columnas
            c = idx % columnas
            nuevo_r[f][c] = 0
            nuevo_g[f][c] = 0
            nuevo_b[f][c] = 0

        for idx in indices[num_pimienta:]:
            f = idx // columnas
            c = idx % columnas
            nuevo_r[f][c] = 255
            nuevo_g[f][c] = 255
            nuevo_b[f][c] = 255
    else:
        umbral_pimienta = densidad / 2.0
        umbral_sal = densidad
        for f in range(filas):
            rf, gf, bf = nuevo_r[f], nuevo_g[f], nuevo_b[f]
            for c in range(columnas):
                p = random.random()
                if p < umbral_pimienta:
                    rf[c] = 0
                    gf[c] = 0
                    bf[c] = 0
                elif p < umbral_sal:
                    rf[c] = 255
                    gf[c] = 255
                    bf[c] = 255

    return nuevo_r, nuevo_g, nuevo_b


def aplicar_mascara_quemada(canal_r, canal_g, canal_b, filas, columnas, modo="enfasis", factor_a=1.0):
    canales_salida = []

    for matriz in (canal_r, canal_g, canal_b):
        matriz_pad = agregar_padding(matriz, filas, columnas, 1, 1)
        salida_canal = crear_matriz(filas, columnas)

        for f in range(filas):
            for c in range(columnas):
                acumulador = 0.0
                for i in range(3):
                    for j in range(3):
                        pixel = matriz_pad[f + i][c + j]
                        peso = MASCARA_QUEMADA_3X3[i][j]
                        acumulador += pixel * peso

                if modo == "laplaciano":
                    nuevo_val = abs(acumulador)
                else:
                                                                                                           
                    detalle_alta_frec = acumulador / 8.0
                    pixel_ifft = matriz[f][c]
                    nuevo_val = pixel_ifft + (float(factor_a) * detalle_alta_frec)

                salida_canal[f][c] = formato_uint8(nuevo_val)

        canales_salida.append(salida_canal)

    return canales_salida[0], canales_salida[1], canales_salida[2], filas, columnas


def calcular_fft2_imagen(canal_r, canal_g, canal_b, filas, columnas):
    arr_r = np.array(canal_r, dtype=np.float64)
    arr_g = np.array(canal_g, dtype=np.float64)
    arr_b = np.array(canal_b, dtype=np.float64)

    arr_gray = 0.299 * arr_r + 0.587 * arr_g + 0.114 * arr_b

    F_r = np.fft.fftshift(np.fft.fft2(arr_r))
    F_g = np.fft.fftshift(np.fft.fft2(arr_g))
    F_b = np.fft.fftshift(np.fft.fft2(arr_b))
    F_gray = np.fft.fftshift(np.fft.fft2(arr_gray))

    return F_r, F_g, F_b, F_gray


def generar_espectro_visual(F_complejo, modo="magnitud"):
    modo = modo.lower()
    if "real" in modo:
        datos = np.real(F_complejo)
    elif "imag" in modo:
        datos = np.imag(F_complejo)
    elif "fase" in modo:
        datos = np.angle(F_complejo)
    else:
        mag = np.abs(F_complejo)
        datos = np.log1p(mag)

    d_min = float(np.min(datos))
    d_max = float(np.max(datos))

    if d_max > d_min:
        norm = ((datos - d_min) / (d_max - d_min) * 255.0).astype(np.uint8)
    else:
        norm = np.zeros(datos.shape, dtype=np.uint8)

    return norm


def generar_mascara_filtro(filas, columnas, tipo_pasa="bajos", d0=40.0, **kwargs):
\
\
\
\
       
    d0 = max(0.5, float(d0))
    u = np.arange(filas) - (filas // 2)
    v = np.arange(columnas) - (columnas // 2)
    U, V = np.meshgrid(u, v, indexing="ij")
    D = np.sqrt(U**2 + V**2)

    H_lp = (D <= d0).astype(np.float64)

    if "alto" in str(tipo_pasa).lower():
        H = 1.0 - H_lp
    else:
        H = H_lp

    return H


def filtrar_frecuencia_rgb(canal_r, canal_g, canal_b, filas, columnas, tipo_pasa="bajos", d0=40.0, **kwargs):
    H = generar_mascara_filtro(filas, columnas, tipo_pasa, d0=d0)

    canales_out = []
    for ch in [canal_r, canal_g, canal_b]:
        arr_ch = np.array(ch, dtype=np.float64)
        F_ch = np.fft.fftshift(np.fft.fft2(arr_ch))
        G_ch = F_ch * H
        g_ch = np.real(np.fft.ifft2(np.fft.ifftshift(G_ch)))
        g_uint8 = np.clip(np.round(g_ch), 0, 255).astype(np.uint8)
        canales_out.append([bytearray(fila) for fila in g_uint8])

    return canales_out[0], canales_out[1], canales_out[2], H
