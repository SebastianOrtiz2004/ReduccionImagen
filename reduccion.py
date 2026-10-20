import math

MASCARA_GAUSSIANA_3X3 = [
    [1, 2, 1],
    [2, 8, 2],
    [1, 2, 1]
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


def obtener_moda(lista):
    mejor_valor = lista[0]
    max_repeticiones = 0
    n = len(lista)
    for i in range(n):
        actual = lista[i]
        conteo = 0
        for j in range(n):
            if lista[j] == actual:
                conteo += 1
        if conteo > max_repeticiones:
            max_repeticiones = conteo
            mejor_valor = actual
    return mejor_valor


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
    if tipo_filtro == "Máscara (Convolución)":
        for i in range(tam_f):
            for j in range(tam_c):
                suma_pesos += mascara[i][j]
        if suma_pesos == 0:
            suma_pesos = 1.0

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
                matriz_salida[f_out][c_out] = obtener_moda(vecinos)

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


if __name__ == "__main__":
    from interfaz import iniciar_interfaz
    iniciar_interfaz()