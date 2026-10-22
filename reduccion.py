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


def procesar_imagen_rgb(canal_r, canal_g, canal_b, filas, columnas, mascara, tam_f, tam_c, tipo_filtro, salto_f, salto_c, usar_pad):
    r_res, n_filas, n_cols = procesar_canal(canal_r, filas, columnas, mascara, tam_f, tam_c, tipo_filtro, salto_f, salto_c, usar_pad)
    g_res, _, _ = procesar_canal(canal_g, filas, columnas, mascara, tam_f, tam_c, tipo_filtro, salto_f, salto_c, usar_pad)
    b_res, _, _ = procesar_canal(canal_b, filas, columnas, mascara, tam_f, tam_c, tipo_filtro, salto_f, salto_c, usar_pad)
    return r_res, g_res, b_res, n_filas, n_cols


def aplicar_ruido_sal_pimienta(canal_r, canal_g, canal_b, filas, columnas, densidad=0.05):
    """
    Aplica ruido impulsivo de Sal y Pimienta sobre una imagen RGB matricial,
    emulando la función imnoise(I, 'salt & pepper', d) de MATLAB / PDI.

    Parámetros:
      - canal_r, canal_g, canal_b: Listas de bytearray por cada fila y canal de color.
      - filas, columnas: Dimensiones espaciales de la imagen.
      - densidad (d): Proporción de píxeles contaminados (en el rango [0.0, 1.0]).
        * ~d / 2 píxeles reciben 'pimienta' (intensidad mínima 0, píxel negro).
        * ~d / 2 píxeles reciben 'sal' (intensidad máxima 255, píxel blanco).
        * Los píxeles restantes permanecen intactos.

    Retorna:
      - (nuevo_r, nuevo_g, nuevo_b): Nuevas matrices con el ruido aplicado.
    """
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

        # Pimienta (Píxel negro: 0 en todos los canales)
        for idx in indices[:num_pimienta]:
            f = idx // columnas
            c = idx % columnas
            nuevo_r[f][c] = 0
            nuevo_g[f][c] = 0
            nuevo_b[f][c] = 0

        # Sal (Píxel blanco: 255 en todos los canales)
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


# Alias para compatibilidad con la nomenclatura estándar imnoise de MATLAB / PDI
imnoise = aplicar_ruido_sal_pimienta


def binarizar_canales(canal_r, canal_g, canal_b, filas, columnas, umbral=128):
    """
    Binariza los canales RGB evaluando la luminancia Y = 0.299R + 0.587G + 0.114B.
    Si Y >= umbral -> 255 (blanco), si Y < umbral -> 0 (negro).
    Retorna copias binarizadas de las listas de bytearray por cada canal.
    """
    umbral = max(0, min(255, int(umbral)))
    nuevo_r = crear_matriz(filas, columnas)
    nuevo_g = crear_matriz(filas, columnas)
    nuevo_b = crear_matriz(filas, columnas)

    for f in range(filas):
        rf, gf, bf = canal_r[f], canal_g[f], canal_b[f]
        n_rf, n_gf, n_bf = nuevo_r[f], nuevo_g[f], nuevo_b[f]
        for c in range(columnas):
            lum = int(0.299 * rf[c] + 0.587 * gf[c] + 0.114 * bf[c] + 0.5)
            val = 255 if lum >= umbral else 0
            n_rf[c] = val
            n_gf[c] = val
            n_bf[c] = val

    return nuevo_r, nuevo_g, nuevo_b


def aplicar_mascara_quemada(canal_r, canal_g, canal_b, filas, columnas, factor_aclarado=1.0, umbral=128, modo="aclarar"):
    """
    Pipeline exacto:
    1. A partir de la imagen filtrada en frecuencia, calcula la luminancia Y y binariza (0 o 255).
    2. A partir de esa imagen binarizada, aplica la convolución con la máscara quemada [-1, 8]
       hardcodeada en código para extraer los bordes y derivadas de segundo orden.
    3. Vuelve a convertir la imagen a normal (a color y no binarizada), sumando el realce
       de los bordes quemados sobre cada canal original:
         canal_final = clip(canal_normal + factor * (laplaciano_bin / 8.0), 0, 255)
       de modo que la imagen vuelve a ser normal (a color y no binarizada), pero completamente
       aclarada y enfocada, eliminando la borrosidad del filtro pasa-bajos.

    Modos:
      - 'aclarar': Imagen normal a color con el realce de la máscara quemada binarizada.
      - 'binarizada': Solo la imagen binarizada previa (blanco y negro).
      - 'laplaciano': Solo la magnitud de los bordes de la máscara quemada (|laplaciano_bin|).
    """
    arr_r = np.array(canal_r, dtype=np.float64)
    arr_g = np.array(canal_g, dtype=np.float64)
    arr_b = np.array(canal_b, dtype=np.float64)

    # 1. Binarizar a partir de la luminancia de la imagen filtrada
    lum = 0.299 * arr_r + 0.587 * arr_g + 0.114 * arr_b
    bin_img = np.where(lum >= float(umbral), 255.0, 0.0)

    if modo == "binarizada":
        bin_uint8 = bin_img.astype(np.uint8)
        ch_bin = [bytearray(fila) for fila in bin_uint8]
        return ch_bin, ch_bin, ch_bin, filas, columnas

    # 2. Máscara quemada [-1, -1, -1; -1, 8, -1; -1, -1, -1] sobre la imagen binarizada
    mask = np.array([
        [-1.0, -1.0, -1.0],
        [-1.0,  8.0, -1.0],
        [-1.0, -1.0, -1.0]
    ], dtype=np.float64)

    bin_pad = np.pad(bin_img, pad_width=1, mode='edge')
    lap_bin = (
        mask[0, 0] * bin_pad[0:filas, 0:columnas] +
        mask[0, 1] * bin_pad[0:filas, 1:columnas+1] +
        mask[0, 2] * bin_pad[0:filas, 2:columnas+2] +
        mask[1, 0] * bin_pad[1:filas+1, 0:columnas] +
        mask[1, 1] * bin_pad[1:filas+1, 1:columnas+1] +
        mask[1, 2] * bin_pad[1:filas+1, 2:columnas+2] +
        mask[2, 0] * bin_pad[2:filas+2, 0:columnas] +
        mask[2, 1] * bin_pad[2:filas+2, 1:columnas+1] +
        mask[2, 2] * bin_pad[2:filas+2, 2:columnas+2]
    )

    if modo == "laplaciano":
        res_lap = np.clip(np.abs(lap_bin), 0, 255).astype(np.uint8)
        ch_lap = [bytearray(fila) for fila in res_lap]
        return ch_lap, ch_lap, ch_lap, filas, columnas

    # 3. Volver a convertir a imagen normal a color, sumándole los bordes quemados
    impulso_bordes = (lap_bin / 8.0) * float(factor_aclarado)

    r_out = np.clip(np.round(arr_r + impulso_bordes), 0, 255).astype(np.uint8)
    g_out = np.clip(np.round(arr_g + impulso_bordes), 0, 255).astype(np.uint8)
    b_out = np.clip(np.round(arr_b + impulso_bordes), 0, 255).astype(np.uint8)

    ch_r = [bytearray(fila) for fila in r_out]
    ch_g = [bytearray(fila) for fila in g_out]
    ch_b = [bytearray(fila) for fila in b_out]

    return ch_r, ch_g, ch_b, filas, columnas


def calcular_fft2_imagen(canal_r, canal_g, canal_b, filas, columnas):
    """
    Calcula la Transformada Rápida de Fourier 2D (FFT2) centrada (fftshift)
    para cada canal de color RGB y para la imagen en escala de grises (luminancia).

    Retorna:
      - F_r, F_g, F_b: matrices complejas 2D centradas de cada canal.
      - F_gray: matriz compleja 2D centrada de la luminancia (escala de grises).
    """
    arr_r = np.array(canal_r, dtype=np.float64)
    arr_g = np.array(canal_g, dtype=np.float64)
    arr_b = np.array(canal_b, dtype=np.float64)

    # Luminancia estándar NTSC / sRGB: Y = 0.299 R + 0.587 G + 0.114 B
    arr_gray = 0.299 * arr_r + 0.587 * arr_g + 0.114 * arr_b

    F_r = np.fft.fftshift(np.fft.fft2(arr_r))
    F_g = np.fft.fftshift(np.fft.fft2(arr_g))
    F_b = np.fft.fftshift(np.fft.fft2(arr_b))
    F_gray = np.fft.fftshift(np.fft.fft2(arr_gray))

    return F_r, F_g, F_b, F_gray


def generar_espectro_visual(F_complejo, modo="magnitud"):
    """
    Genera una imagen visualizable (matriz uint8 2D de 0 a 255) a partir de una matriz compleja FFT.

    Modos disponibles:
      - 'magnitud': Escala logarítmica log(1 + |F(u,v)|) normalizada a [0, 255] (estándar PDI).
      - 'real': Parte real R(u,v) normalizada a [0, 255].
      - 'imaginario': Parte imaginaria I(u,v) normalizada a [0, 255].
      - 'fase': Ángulo de fase arctan2(I, R) normalizado a [0, 255].
    """
    modo = modo.lower()
    if "real" in modo:
        datos = np.real(F_complejo)
    elif "imag" in modo:
        datos = np.imag(F_complejo)
    elif "fase" in modo:
        datos = np.angle(F_complejo)
    else:  # magnitud
        mag = np.abs(F_complejo)
        datos = np.log1p(mag)

    d_min = float(np.min(datos))
    d_max = float(np.max(datos))

    if d_max > d_min:
        norm = ((datos - d_min) / (d_max - d_min) * 255.0).astype(np.uint8)
    else:
        norm = np.zeros(datos.shape, dtype=np.uint8)

    return norm


def generar_mascara_filtro(filas, columnas, tipo_pasa="bajos", modelo="gaussiano", d0=40.0, orden=2):
    """
    Construye la función de transferencia del filtro centrado H(u, v) en el dominio frecuencial.

    Parámetros:
      - filas, columnas: dimensiones de la imagen.
      - tipo_pasa: 'bajos' (Low-Pass) o 'altos' (High-Pass).
      - modelo: 'gaussiano', 'butterworth', o 'ideal'.
      - d0: frecuencia de corte D0 (radio en píxeles frecuenciales).
      - orden: orden n para el filtro Butterworth.

    Retorna:
      - H: matriz numpy 2D float64 con valores en [0.0, 1.0].
    """
    d0 = max(0.5, float(d0))
    u = np.arange(filas) - (filas // 2)
    v = np.arange(columnas) - (columnas // 2)
    U, V = np.meshgrid(u, v, indexing="ij")
    D = np.sqrt(U**2 + V**2)

    mod = modelo.lower()
    if "ideal" in mod:
        H_lp = (D <= d0).astype(np.float64)
    elif "butter" in mod:
        orden = max(1, int(orden))
        with np.errstate(divide="ignore", invalid="ignore"):
            H_lp = 1.0 / (1.0 + (D / d0)**(2 * orden))
    else:  # gaussiano por defecto
        H_lp = np.exp(-(D**2) / (2.0 * (d0**2)))

    if "alto" in tipo_pasa.lower():
        H = 1.0 - H_lp
    else:
        H = H_lp

    return H


def filtrar_frecuencia_rgb(canal_r, canal_g, canal_b, filas, columnas, tipo_pasa="bajos", modelo="gaussiano", d0=40.0, orden=2):
    """
    Aplica filtrado en el dominio de la frecuencia 2D mediante convolución multiplicativa:
    G(u, v) = H(u, v) * F(u, v) y antitransformada IFFT2.

    Retorna:
      - r_filt, g_filt, b_filt: listas de bytearray por canal con los píxeles filtrados.
      - H: máscara de filtro aplicada.
    """
    H = generar_mascara_filtro(filas, columnas, tipo_pasa, modelo, d0, orden)

    canales_out = []
    for ch in [canal_r, canal_g, canal_b]:
        arr_ch = np.array(ch, dtype=np.float64)
        F_ch = np.fft.fftshift(np.fft.fft2(arr_ch))
        G_ch = F_ch * H
        g_ch = np.real(np.fft.ifft2(np.fft.ifftshift(G_ch)))
        g_uint8 = np.clip(np.round(g_ch), 0, 255).astype(np.uint8)
        canales_out.append([bytearray(fila) for fila in g_uint8])

    return canales_out[0], canales_out[1], canales_out[2], H


if __name__ == "__main__":
    from interfaz import iniciar_interfaz
    iniciar_interfaz()