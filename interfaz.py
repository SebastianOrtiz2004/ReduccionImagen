import math
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
from PIL import Image, ImageTk

from reduccion import (
    formato_uint8,
    crear_matriz,
    agregar_padding,
    ordenar_elementos,
    obtener_moda,
    procesar_canal,
    calcular_matriz_gaussiana,
    MASCARA_GAUSSIANA_3X3,
    MASCARA_QUEMADA_3X3,
    aplicar_ruido_sal_pimienta,
    calcular_fft2_imagen,
    generar_espectro_visual,
    generar_mascara_filtro,
    filtrar_frecuencia_rgb,
    aplicar_mascara_quemada,
)

ancho_orig = 0
alto_orig = 0
canal_rojo = []
canal_verde = []
canal_azul = []
canal_rojo_puro = []
canal_verde_puro = []
canal_azul_puro = []
densidad_ruido_actual = 0.0
entrada_ruido = None
entrada_ruido_frec = None

var_frec_quemada = None
var_modo_acentuado = None
scale_factor_a = None
lbl_tit_frec_filtrada = None

F_r_actual = None
F_g_actual = None
F_b_actual = None
F_gray_actual = None
H_actual = None

r_frec_filtrado = None
g_frec_filtrado = None
b_frec_filtrado = None

foto_frec_orig_tk = None
foto_frec_ruido_tk = None
foto_frec_fft_tk = None
foto_frec_filtrada_tk = None

frec_fft_w_disp = 0
frec_fft_h_disp = 0

lbl_frec_info_dim = None
combo_tipo_pasa = None
combo_modelo_filtro = None
scale_d0 = None
entry_d0 = None
combo_modo_espectro = None

lbl_info_frec_orig = None
lbl_disp_frec_orig = None
lbl_info_frec_ruido = None
lbl_disp_frec_ruido = None
lbl_info_frec_fft = None
canvas_disp_frec_fft = None
lbl_info_frec_filtrada = None
lbl_disp_frec_filtrada = None

lbl_insp_uv = None
lbl_insp_radio = None
lbl_insp_real = None
lbl_insp_imag = None
lbl_insp_mag = None
lbl_insp_fase = None

MAX_W_QUAD = 440
MAX_H_QUAD = 250

vista_orig_tk = None
vista_res_tk = None
entradas_matriz = []

nombre_mascara_activa = "Gaussiana Ejemplo"
mascara_aplicada_guardada = []
nombre_filtro_aplicado_guardado = ""
suma_pesos_guardada = 1.0

img_resultado_pil_nativa = None
zoom_factor = 1.0
pan_x = 0.0
pan_y = 0.0
drag_start_x = 0
drag_start_y = 0
drag_start_pan_x = 0.0
drag_start_pan_y = 0.0
canvas_resultado = None
img_canvas_id = None
lbl_zoom_val = None

lbl_tipo_mascara_activa = None
lbl_info_resultado_mascara = None
btn_ver_mascara = None
lbl_img_original = None
lbl_info_original_dim = None
lbl_img_resultado = None
lbl_info_resultado_dim = None
lbl_info_resultado_detalles = None
entrada_tamano = None
combo_filtro = None
entrada_salto_f = None
entrada_salto_c = None
var_padding = None
frame_cuadricula = None
ventana = None

MAX_ANCHO_PREVIEW = 440
MAX_ALTO_PREVIEW = 460


def generar_foto_preview(matriz_r, matriz_g, matriz_b, filas, columnas):
    buffer_rgb = bytearray(filas * columnas * 3)
    pos = 0
    for f in range(filas):
        rf, gf, bf = matriz_r[f], matriz_g[f], matriz_b[f]
        for c in range(columnas):
            buffer_rgb[pos] = rf[c]
            buffer_rgb[pos + 1] = gf[c]
            buffer_rgb[pos + 2] = bf[c]
            pos += 3

    img_nativa = Image.frombytes("RGB", (columnas, filas), bytes(buffer_rgb))

    escala = min(MAX_ANCHO_PREVIEW / ancho_orig, MAX_ALTO_PREVIEW / alto_orig, 1.0)
    ancho_disp = max(1, int(columnas * escala))
    alto_disp = max(1, int(filas * escala))

    img_vista = img_nativa.copy()
    img_vista.thumbnail((ancho_disp, alto_disp), Image.Resampling.LANCZOS)
    return ImageTk.PhotoImage(img_vista)


def generar_foto_preview_rgb(matriz_r, matriz_g, matriz_b, filas, columnas, max_w=440, max_h=250):
    buffer_rgb = bytearray(filas * columnas * 3)
    pos = 0
    for f in range(filas):
        rf, gf, bf = matriz_r[f], matriz_g[f], matriz_b[f]
        for c in range(columnas):
            buffer_rgb[pos] = rf[c]
            buffer_rgb[pos + 1] = gf[c]
            buffer_rgb[pos + 2] = bf[c]
            pos += 3

    img_nativa = Image.frombytes("RGB", (columnas, filas), bytes(buffer_rgb))
    escala = min(max_w / max(1, columnas), max_h / max(1, filas), 1.0)
    w_disp = max(1, int(columnas * escala))
    h_disp = max(1, int(filas * escala))

    img_vista = img_nativa.resize((w_disp, h_disp), Image.Resampling.LANCZOS)
    return ImageTk.PhotoImage(img_vista), w_disp, h_disp


def generar_foto_preview_array(arr_2d, max_w=440, max_h=250):
    img_pil = Image.fromarray(arr_2d, mode="L").convert("RGB")
    w, h = img_pil.size
    escala = min(max_w / max(1, w), max_h / max(1, h), 1.0)
    w_disp = max(1, int(w * escala))
    h_disp = max(1, int(h * escala))

    img_vista = img_pil.resize((w_disp, h_disp), Image.Resampling.LANCZOS)
    return ImageTk.PhotoImage(img_vista), w_disp, h_disp


def actualizar_render_zoom():
    global vista_res_tk, img_canvas_id
    if img_resultado_pil_nativa is None or canvas_resultado is None:
        return

    n_cols, n_filas = img_resultado_pil_nativa.size

    if ancho_orig > 0 and alto_orig > 0:
        escala_base = min(MAX_ANCHO_PREVIEW / ancho_orig, MAX_ALTO_PREVIEW / alto_orig, 1.0)
    else:
        escala_base = 1.0

    base_w = max(1, int(n_cols * escala_base))
    base_h = max(1, int(n_filas * escala_base))

    target_w = max(1, int(base_w * zoom_factor))
    target_h = max(1, int(base_h * zoom_factor))

    target_w = min(target_w, 4000)
    target_h = min(target_h, 4000)

    img_render = img_resultado_pil_nativa.resize((target_w, target_h), Image.Resampling.LANCZOS)
    vista_res_tk = ImageTk.PhotoImage(img_render)

    canvas_resultado.delete("all")
    cx = (MAX_ANCHO_PREVIEW // 2) + pan_x
    cy = (MAX_ALTO_PREVIEW // 2) + pan_y
    img_canvas_id = canvas_resultado.create_image(cx, cy, anchor="center", image=vista_res_tk)
    canvas_resultado.config(cursor="fleur")

    if lbl_zoom_val is not None and lbl_zoom_val.winfo_exists():
        lbl_zoom_val.config(text=f"Zoom: {zoom_factor:.2f}×")

    if lbl_info_resultado_dim is not None and lbl_info_resultado_dim.winfo_exists():
        if abs(zoom_factor - 1.0) < 0.01:
            lbl_info_resultado_dim.config(text=f"{n_filas} x {n_cols} px  (1.0× Tamaño Reducido)")
        else:
            lbl_info_resultado_dim.config(text=f"{n_filas} x {n_cols} px  |  Zoom {zoom_factor:.2f}×")


def aplicar_delta_zoom(delta_factor, center_x=None, center_y=None):
    global zoom_factor, pan_x, pan_y
    if img_resultado_pil_nativa is None:
        return

    nuevo_zoom = zoom_factor * delta_factor
    nuevo_zoom = max(0.25, min(15.0, nuevo_zoom))
    if abs(nuevo_zoom - zoom_factor) < 1e-4:
        return

    if center_x is None:
        center_x = MAX_ANCHO_PREVIEW / 2.0
    if center_y is None:
        center_y = MAX_ALTO_PREVIEW / 2.0

    cx_ant = (MAX_ANCHO_PREVIEW / 2.0) + pan_x
    cy_ant = (MAX_ALTO_PREVIEW / 2.0) + pan_y

    ratio = nuevo_zoom / zoom_factor
    cx_nuevo = center_x - (center_x - cx_ant) * ratio
    cy_nuevo = center_y - (center_y - cy_ant) * ratio

    pan_x = cx_nuevo - (MAX_ANCHO_PREVIEW / 2.0)
    pan_y = cy_nuevo - (MAX_ALTO_PREVIEW / 2.0)
    zoom_factor = nuevo_zoom

    actualizar_render_zoom()


def on_scroll_zoom(event):
    if img_resultado_pil_nativa is None:
        return
    factor = 1.25 if event.delta > 0 else (1.0 / 1.25)
    aplicar_delta_zoom(factor, event.x, event.y)


def on_drag_inicio(event):
    global drag_start_x, drag_start_y, drag_start_pan_x, drag_start_pan_y
    drag_start_x = event.x
    drag_start_y = event.y
    drag_start_pan_x = pan_x
    drag_start_pan_y = pan_y


def on_drag_mover(event):
    global pan_x, pan_y
    if img_resultado_pil_nativa is None or img_canvas_id is None:
        return
    pan_x = drag_start_pan_x + (event.x - drag_start_x)
    pan_y = drag_start_pan_y + (event.y - drag_start_y)
    cx = (MAX_ANCHO_PREVIEW // 2) + pan_x
    cy = (MAX_ALTO_PREVIEW // 2) + pan_y
    canvas_resultado.coords(img_canvas_id, cx, cy)


def resetear_zoom(event=None):
    global zoom_factor, pan_x, pan_y
    if img_resultado_pil_nativa is None:
        return
    zoom_factor = 1.0
    pan_x = 0.0
    pan_y = 0.0
    actualizar_render_zoom()


def actualizar_indicador_mascara(nombre=None):
    global nombre_mascara_activa
    if nombre:
        nombre_mascara_activa = nombre
    try:
        if lbl_tipo_mascara_activa is not None and lbl_tipo_mascara_activa.winfo_exists():
            lbl_tipo_mascara_activa.config(text=f"Máscara Activa: {nombre_mascara_activa}")
    except Exception:
        pass


def crear_cuadricula_vacia():
    global entradas_matriz
    try:
        tam = int(entrada_tamano.get())
        if tam < 3 or tam % 2 == 0:
            messagebox.showwarning("Atención", "El tamaño de la máscara debe ser impar (3, 5, 7, ...).")
            return
    except ValueError:
        messagebox.showerror("Error", "Ingresa un número entero impar.")
        return

    for w in frame_cuadricula.winfo_children():
        w.destroy()

    entradas_matriz = []
    for i in range(tam):
        fila_cajas = []
        for j in range(tam):
            caja = tk.Entry(frame_cuadricula, width=6, justify="center", font=("Arial", 8))
            caja.grid(row=i, column=j, padx=1, pady=1)
            caja.insert(0, "1")
            fila_cajas.append(caja)
        entradas_matriz.append(fila_cajas)

    actualizar_indicador_mascara(f"Personalizada ({tam}x{tam})")


def calcular_gaussiana_en_cuadricula():
    crear_cuadricula_vacia()
    tam = len(entradas_matriz)
    pesos_normalizados = calcular_matriz_gaussiana(tam)

    for i in range(tam):
        for j in range(tam):
            entradas_matriz[i][j].delete(0, tk.END)
            entradas_matriz[i][j].insert(0, f"{pesos_normalizados[i][j]:.4f}")

    combo_filtro.set("Máscara (Convolución)")
    actualizar_indicador_mascara("Gaussiana")


def cargar_preset_gaussiana_clase():
    entrada_tamano.delete(0, tk.END)
    entrada_tamano.insert(0, "3")
    crear_cuadricula_vacia()
    for i in range(3):
        for j in range(3):
            entradas_matriz[i][j].delete(0, tk.END)
            entradas_matriz[i][j].insert(0, str(MASCARA_GAUSSIANA_3X3[i][j]))
    combo_filtro.set("Máscara (Convolución)")
    actualizar_indicador_mascara("Gaussiana Ejemplo")


def cargar_preset_media():
    crear_cuadricula_vacia()
    for fila in entradas_matriz:
        for caja in fila:
            caja.delete(0, tk.END)
            caja.insert(0, "1")
    combo_filtro.set("Máscara (Convolución)")
    actualizar_indicador_mascara("Media")


def cargar_preset_acentuado():
    entrada_tamano.delete(0, tk.END)
    entrada_tamano.insert(0, "3")
    crear_cuadricula_vacia()
    for i in range(3):
        for j in range(3):
            entradas_matriz[i][j].delete(0, tk.END)
            entradas_matriz[i][j].insert(0, str(MASCARA_QUEMADA_3X3[i][j]))
    combo_filtro.set("Máscara (Convolución)")
    actualizar_indicador_mascara("Acentuado [-1, 8]")


def on_cambio_filtro(*args):
    tipo = combo_filtro.get()
    if tipo == "Mediana":
        actualizar_indicador_mascara("Mediana")
    elif tipo == "Moda":
        actualizar_indicador_mascara("Moda")
    else:
        actualizar_indicador_mascara()


def aplicar_y_mostrar_ruido(densidad):
    global canal_rojo, canal_verde, canal_azul, densidad_ruido_actual, vista_orig_tk
    global img_resultado_pil_nativa, zoom_factor, pan_x, pan_y

    if not canal_rojo_puro or alto_orig == 0 or ancho_orig == 0:
        return

    densidad_ruido_actual = max(0.0, min(1.0, float(densidad)))

    if entrada_ruido is not None and entrada_ruido.winfo_exists():
        entrada_ruido.delete(0, tk.END)
        entrada_ruido.insert(0, str(round(densidad_ruido_actual, 3)))

    if entrada_ruido_frec is not None and entrada_ruido_frec.winfo_exists():
        entrada_ruido_frec.delete(0, tk.END)
        entrada_ruido_frec.insert(0, str(round(densidad_ruido_actual, 3)))

    if densidad_ruido_actual > 0.0:
        canal_rojo, canal_verde, canal_azul = aplicar_ruido_sal_pimienta(
            canal_rojo_puro, canal_verde_puro, canal_azul_puro,
            alto_orig, ancho_orig, densidad_ruido_actual
        )
    else:
        canal_rojo = [bytearray(fila) for fila in canal_rojo_puro]
        canal_verde = [bytearray(fila) for fila in canal_verde_puro]
        canal_azul = [bytearray(fila) for fila in canal_azul_puro]

    vista_orig_tk = generar_foto_preview(canal_rojo, canal_verde, canal_azul, alto_orig, ancho_orig)
    lbl_img_original.config(image=vista_orig_tk, text="")

    if densidad_ruido_actual > 0:
        porcentaje = densidad_ruido_actual * 100.0
        lbl_info_original_dim.config(
            text=f"{alto_orig} x {ancho_orig} px  |  Ruido: {porcentaje:.1f}% (d={densidad_ruido_actual:.2f})"
        )
    else:
        lbl_info_original_dim.config(text=f"{alto_orig} x {ancho_orig} px  (Sin ruido)")

    img_resultado_pil_nativa = None
    zoom_factor = 1.0
    pan_x = 0.0
    pan_y = 0.0
    if canvas_resultado is not None and canvas_resultado.winfo_exists():
        canvas_resultado.delete("all")
        canvas_resultado.config(cursor="arrow")
        canvas_resultado.create_text(
            MAX_ANCHO_PREVIEW // 2, MAX_ALTO_PREVIEW // 2,
            text="Sin resultado\n\nAjusta los parámetros y presiona 'Procesar y Reducir'",
            fill="#555555", font=("Arial", 10), justify="center", tags="placeholder"
        )
    if lbl_info_resultado_dim is not None and lbl_info_resultado_dim.winfo_exists():
        lbl_info_resultado_dim.config(text="Sin procesar")
    if lbl_zoom_val is not None and lbl_zoom_val.winfo_exists():
        lbl_zoom_val.config(text="Zoom: 1.00×")
    if btn_ver_mascara is not None and btn_ver_mascara.winfo_exists():
        btn_ver_mascara.config(state=tk.DISABLED)

    actualizar_tab_frecuencia()


def on_click_aplicar_ruido_frec():
    if not canal_rojo_puro:
        messagebox.showinfo("Aviso", "Primero carga una imagen.")
        return
    try:
        val = float(entrada_ruido_frec.get())
        if val < 0.0 or val > 1.0:
            raise ValueError
    except ValueError:
        messagebox.showerror("Error", "El parámetro de ruido debe ser un número entre 0.0 y 1.0.")
        return
    aplicar_y_mostrar_ruido(val)


def actualizar_inspector_fft(f, c):
    if F_gray_actual is None or alto_orig == 0 or ancho_orig == 0:
        return

    f = max(0, min(alto_orig - 1, int(f)))
    c = max(0, min(ancho_orig - 1, int(c)))

    f0 = alto_orig // 2
    c0 = ancho_orig // 2
    u = f - f0
    v = c - c0
    dist = math.sqrt(u**2 + v**2)

    coef = F_gray_actual[f, c]
    real_val = float(np.real(coef))
    imag_val = float(np.imag(coef))
    mag_val = float(np.abs(coef))
    fase_deg = float(np.angle(coef, deg=True))

    es_dc = (u == 0 and v == 0)
    str_dc = " [DC]" if es_dc else ""

    if lbl_insp_uv and lbl_insp_uv.winfo_exists():
        lbl_insp_uv.config(text=f"Frecuencia (u, v): ({u:+d}, {v:+d})  {str_dc}")
    if lbl_insp_radio and lbl_insp_radio.winfo_exists():
        lbl_insp_radio.config(text=f"Radio D(u, v): {dist:.2f} px")
    if lbl_insp_real and lbl_insp_real.winfo_exists():
        lbl_insp_real.config(text=f"Parte Real: {real_val:+.3e}")
    if lbl_insp_imag and lbl_insp_imag.winfo_exists():
        lbl_insp_imag.config(text=f"Parte Imag: {imag_val:+.3e} j")
    if lbl_insp_mag and lbl_insp_mag.winfo_exists():
        lbl_insp_mag.config(text=f"Magnitud |F|: {mag_val:.3e}")
    if lbl_insp_fase and lbl_insp_fase.winfo_exists():
        lbl_insp_fase.config(text=f"Fase φ: {fase_deg:+.1f}°")


def on_hover_espectro_fft(event):
    if F_gray_actual is None or frec_fft_w_disp <= 0 or frec_fft_h_disp <= 0:
        return

    cw = canvas_disp_frec_fft.winfo_width() or MAX_W_QUAD
    ch = canvas_disp_frec_fft.winfo_height() or MAX_H_QUAD

    offset_x = (cw - frec_fft_w_disp) // 2
    offset_y = (ch - frec_fft_h_disp) // 2

    rel_x = event.x - offset_x
    rel_y = event.y - offset_y

    if 0 <= rel_x < frec_fft_w_disp and 0 <= rel_y < frec_fft_h_disp:
        c = int(rel_x * (ancho_orig - 1) / max(1, frec_fft_w_disp - 1))
        f = int(rel_y * (alto_orig - 1) / max(1, frec_fft_h_disp - 1))
        actualizar_inspector_fft(f, c)


def on_leave_espectro_fft(event):
    if F_gray_actual is not None and alto_orig > 0 and ancho_orig > 0:
        actualizar_inspector_fft(alto_orig // 2, ancho_orig // 2)


def actualizar_render_espectro_fft():
    global foto_frec_fft_tk, frec_fft_w_disp, frec_fft_h_disp
    if F_gray_actual is None or canvas_disp_frec_fft is None or not canvas_disp_frec_fft.winfo_exists():
        return

    modo = combo_modo_espectro.get() if combo_modo_espectro else "Magnitud Logarítmica log(1+|F|)"

    if "Máscara" in modo:
        if H_actual is not None:
            arr_visual = (H_actual * 255.0).astype(np.uint8)
        else:
            d0_val = float(scale_d0.get()) if scale_d0 else 40.0
            tipo_val = "bajos" if (combo_tipo_pasa and "Bajos" in combo_tipo_pasa.get()) else "altos"
            arr_visual = (generar_mascara_filtro(alto_orig, ancho_orig, tipo_val, d0=d0_val) * 255.0).astype(np.uint8)
        if lbl_info_frec_fft and lbl_info_frec_fft.winfo_exists():
            lbl_info_frec_fft.config(text=f"Máscara Filtro H(u,v) | {alto_orig} x {ancho_orig} px")
    elif "Real" in modo:
        arr_visual = generar_espectro_visual(F_gray_actual, "real")
        if lbl_info_frec_fft and lbl_info_frec_fft.winfo_exists():
            lbl_info_frec_fft.config(text=f"Parte Real Re{{F(u,v)}} | {alto_orig} x {ancho_orig} px")
    elif "Imag" in modo:
        arr_visual = generar_espectro_visual(F_gray_actual, "imaginario")
        if lbl_info_frec_fft and lbl_info_frec_fft.winfo_exists():
            lbl_info_frec_fft.config(text=f"Parte Imaginaria Im{{F(u,v)}} | {alto_orig} x {ancho_orig} px")
    elif "Fase" in modo:
        arr_visual = generar_espectro_visual(F_gray_actual, "fase")
        if lbl_info_frec_fft and lbl_info_frec_fft.winfo_exists():
            lbl_info_frec_fft.config(text=f"Ángulo de Fase φ(u,v) | {alto_orig} x {ancho_orig} px")
    else:
        arr_visual = generar_espectro_visual(F_gray_actual, "magnitud")
        if lbl_info_frec_fft and lbl_info_frec_fft.winfo_exists():
            lbl_info_frec_fft.config(text=f"Espectro Magnitud log(1+|F|) | {alto_orig} x {ancho_orig} px")

    foto_frec_fft_tk, frec_fft_w_disp, frec_fft_h_disp = generar_foto_preview_array(
        arr_visual, max_w=MAX_W_QUAD, max_h=MAX_H_QUAD
    )

    canvas_disp_frec_fft.delete("all")
    cw = canvas_disp_frec_fft.winfo_width() or MAX_W_QUAD
    ch = canvas_disp_frec_fft.winfo_height() or MAX_H_QUAD
    canvas_disp_frec_fft.create_image(cw // 2, ch // 2, anchor="center", image=foto_frec_fft_tk)

    actualizar_inspector_fft(alto_orig // 2, ancho_orig // 2)


def on_cambio_modo_espectro(*args):
    actualizar_render_espectro_fft()


def ejecutar_filtro_frecuencia():
    global r_frec_filtrado, g_frec_filtrado, b_frec_filtrado, H_actual

    if alto_orig == 0 or ancho_orig == 0 or not canal_rojo:
        return

    tipo_pasa_str = combo_tipo_pasa.get() if combo_tipo_pasa else "Pasa Bajos"
    tipo_pasa = "bajos" if "Bajos" in tipo_pasa_str else "altos"

    try:
        d0 = float(scale_d0.get()) if scale_d0 else 40.0
        if d0 <= 0:
            d0 = 40.0
    except ValueError:
        d0 = 40.0

    r_filt, g_filt, b_filt, H = filtrar_frecuencia_rgb(
        canal_rojo, canal_verde, canal_azul, alto_orig, ancho_orig,
        tipo_pasa=tipo_pasa, d0=d0
    )

    r_frec_filtrado = r_filt
    g_frec_filtrado = g_filt
    b_frec_filtrado = b_filt
    H_actual = H

    if combo_modo_espectro and "Máscara" in combo_modo_espectro.get():
        actualizar_render_espectro_fft()

    actualizar_vista_cuadrante_4()


def actualizar_vista_cuadrante_4():
    global foto_frec_filtrada_tk

    if alto_orig == 0 or ancho_orig == 0 or r_frec_filtrado is None:
        return

    if var_frec_quemada is not None and var_frec_quemada.get():
        modo_actual = var_modo_acentuado.get() if var_modo_acentuado else "enfasis"
        try:
            factor_a = float(scale_factor_a.get()) if scale_factor_a else 1.0
        except Exception:
            factor_a = 1.0

        r_out, g_out, b_out, _, _ = aplicar_mascara_quemada(
            r_frec_filtrado, g_frec_filtrado, b_frec_filtrado,
            alto_orig, ancho_orig,
            modo=modo_actual, factor_a=factor_a
        )

        if modo_actual == "enfasis":
            titulo_txt = "4. Imagen IFFT + Énfasis de Altas Frecuencias (EA)"
            info_txt = f"{alto_orig}x{ancho_orig} px | EA = IFFT + {factor_a:.1f}·(PA/8) | Color e iluminación intactos + Bordes enfocados"
        else:
            titulo_txt = "4. Imagen IFFT + Pasa-Altos Puro [-1, 8] (PA)"
            info_txt = f"{alto_orig}x{ancho_orig} px | Convolución directa con máscara [-1, 8] (Suma = 0, orillas sobre negro)"
    else:
        r_out, g_out, b_out = r_frec_filtrado, g_frec_filtrado, b_frec_filtrado
        tipo_pasa_str = combo_tipo_pasa.get() if combo_tipo_pasa else "Pasa Bajos"
        tipo_corto = "Pasa Bajos" if "Bajos" in tipo_pasa_str else "Pasa Altos"
        d0_val = float(scale_d0.get()) if scale_d0 else 40.0
        titulo_txt = "4. Imagen Filtrada en Frecuencia (IFFT)"
        info_txt = f"{alto_orig}x{ancho_orig} px | {tipo_corto} | D₀={d0_val:.0f} px"

    if lbl_disp_frec_filtrada and lbl_disp_frec_filtrada.winfo_exists():
        foto_frec_filtrada_tk, _, _ = generar_foto_preview_rgb(
            r_out, g_out, b_out, alto_orig, ancho_orig, max_w=MAX_W_QUAD, max_h=MAX_H_QUAD
        )
        lbl_disp_frec_filtrada.config(image=foto_frec_filtrada_tk, text="")

    if lbl_tit_frec_filtrada and lbl_tit_frec_filtrada.winfo_exists():
        lbl_tit_frec_filtrada.config(text=titulo_txt)

    if lbl_info_frec_filtrada and lbl_info_frec_filtrada.winfo_exists():
        lbl_info_frec_filtrada.config(text=info_txt)


def mostrar_matriz_quemada_popup():
    popup = tk.Toplevel(ventana)
    popup.title("Detalles del Filtro de Acentuado")
    popup.geometry("540x440")
    popup.configure(bg="#1a1a1a")
    popup.resizable(False, False)

    modo_actual = var_modo_acentuado.get() if var_modo_acentuado else "enfasis"
    try:
        factor_a = float(scale_factor_a.get()) if scale_factor_a else 1.0
    except Exception:
        factor_a = 1.0

    if modo_actual == "enfasis":
        w_centro = 9.0 * factor_a - 1.0
        suma_pesos = w_centro - 8.0

        tk.Label(popup, text="Énfasis de Altas Frecuencias (High-Boost)", bg="#1a1a1a", fg="#4ec9b0", font=("Arial", 11, "bold")).pack(pady=(12, 2))
        tk.Label(popup, text="Diapositivas 16 a 18: EA = Imagen_IFFT + A · (PA / 8)", bg="#1a1a1a", fg="#ffd700", font=("Arial", 9, "bold")).pack()
        tk.Label(
            popup,
            text=f"Equivalente a una máscara espacial con celda central w = 9A - 1:\n(Para A = {factor_a:.2f} ⟹ Centro = {w_centro:.2f}, Suma de pesos = {suma_pesos:.2f})",
            bg="#1a1a1a", fg="#cccccc", font=("Arial", 9), justify="center"
        ).pack(pady=(2, 6))

        frame_box = tk.Frame(popup, bg="#0d0d0d", padx=10, pady=8)
        frame_box.pack(padx=20, pady=4)
        caja = tk.Text(frame_box, bg="#0d0d0d", fg="#4ec9b0", font=("Consolas", 11, "bold"), width=32, height=4, bd=0)
        caja.pack()
        caja.insert(tk.END, f"  [   -1.00     -1.00    -1.00   ]\n  [   -1.00    {w_centro:6.2f}    -1.00   ]\n  [   -1.00     -1.00    -1.00   ]\n")
        caja.config(state=tk.DISABLED)

        tk.Label(
            popup,
            text="✓ A diferencia del pasa-altos puro que anula las bajas frecuencias dejando el fondo negro,\n"
                 "el énfasis de altas frecuencias conserva la iluminación y colores originales de la IFFT,\n"
                 "restituyendo la nitidez y el enfoque de las orillas (Diapositivas 19c y 19f).",
            bg="#1a1a1a", fg="#9cdcfe", font=("Arial", 8, "italic"), justify="center"
        ).pack(pady=(6, 8))
    else:
        tk.Label(popup, text="Filtro Pasa-Altos Puro [-1, 8] (Diapositiva 14)", bg="#1a1a1a", fg="#ffd700", font=("Arial", 11, "bold")).pack(pady=(12, 4))
        tk.Label(
            popup,
            text="Convolución directa con máscara pasa-altos derivativa 3x3:",
            bg="#1a1a1a", fg="#cccccc", font=("Arial", 9), justify="center"
        ).pack(pady=(0, 6))

        frame_box = tk.Frame(popup, bg="#0d0d0d", padx=10, pady=8)
        frame_box.pack(padx=20, pady=4)
        caja = tk.Text(frame_box, bg="#0d0d0d", fg="#4ec9b0", font=("Consolas", 12, "bold"), width=24, height=4, bd=0)
        caja.pack()
        caja.insert(tk.END, "  [ -1   -1   -1 ]\n  [ -1    8   -1 ]\n  [ -1   -1   -1 ]\n")
        caja.config(state=tk.DISABLED)

        tk.Label(
            popup,
            text="Suma de pesos = 0.\nLas regiones de intensidad constante dan 0 (fondo negro) y solo se aprecian las orillas.",
            bg="#1a1a1a", fg="#9cdcfe", font=("Arial", 8, "italic"), justify="center"
        ).pack(pady=(6, 10))

    tk.Button(popup, text="Cerrar", command=popup.destroy, bg="#333333", fg="white", font=("Arial", 9, "bold"), width=12, relief=tk.FLAT).pack(pady=(0, 10))


def actualizar_tab_frecuencia():
    global foto_frec_orig_tk, foto_frec_ruido_tk
    global F_r_actual, F_g_actual, F_b_actual, F_gray_actual

    if alto_orig == 0 or ancho_orig == 0 or not canal_rojo_puro:
        return

    porcentaje = densidad_ruido_actual * 100.0

    if lbl_disp_frec_orig is not None and lbl_disp_frec_orig.winfo_exists():
        foto_frec_orig_tk, _, _ = generar_foto_preview_rgb(
            canal_rojo_puro, canal_verde_puro, canal_azul_puro,
            alto_orig, ancho_orig, max_w=MAX_W_QUAD, max_h=MAX_H_QUAD
        )
        lbl_disp_frec_orig.config(image=foto_frec_orig_tk, text="")
    if lbl_info_frec_orig is not None and lbl_info_frec_orig.winfo_exists():
        lbl_info_frec_orig.config(text=f"{alto_orig} x {ancho_orig} px (Limpia sin ruido)")

    if lbl_disp_frec_ruido is not None and lbl_disp_frec_ruido.winfo_exists():
        foto_frec_ruido_tk, _, _ = generar_foto_preview_rgb(
            canal_rojo, canal_verde, canal_azul,
            alto_orig, ancho_orig, max_w=MAX_W_QUAD, max_h=MAX_H_QUAD
        )
        lbl_disp_frec_ruido.config(image=foto_frec_ruido_tk, text="")
    if lbl_info_frec_ruido is not None and lbl_info_frec_ruido.winfo_exists():
        if densidad_ruido_actual > 0:
            lbl_info_frec_ruido.config(text=f"{alto_orig} x {ancho_orig} px | Ruido: {porcentaje:.1f}% (d={densidad_ruido_actual:.2f})")
        else:
            lbl_info_frec_ruido.config(text=f"{alto_orig} x {ancho_orig} px (Sin ruido)")

    if lbl_frec_info_dim is not None and lbl_frec_info_dim.winfo_exists():
        lbl_frec_info_dim.config(text=f"Imagen: {alto_orig}x{ancho_orig} px | Ruido: {porcentaje:.1f}%")

    F_r_actual, F_g_actual, F_b_actual, F_gray_actual = calcular_fft2_imagen(
        canal_rojo, canal_verde, canal_azul, alto_orig, ancho_orig
    )

    actualizar_render_espectro_fft()
    ejecutar_filtro_frecuencia()


def on_scale_d0_change(val):
    try:
        fval = float(val)
        if entry_d0 is not None and entry_d0.winfo_exists():
            entry_d0.delete(0, tk.END)
            entry_d0.insert(0, f"{fval:.0f}")
    except Exception:
        pass


def on_entry_d0_change(event=None):
    try:
        val = float(entry_d0.get())
        if scale_d0 is not None and scale_d0.winfo_exists():
            scale_d0.set(val)
    except Exception:
        pass


def on_click_aplicar_ruido():
    if not canal_rojo_puro:
        messagebox.showinfo("Aviso", "Primero carga una imagen.")
        return
    try:
        val = float(entrada_ruido.get())
        if val < 0.0 or val > 1.0:
            raise ValueError
    except ValueError:
        messagebox.showerror("Error", "El parámetro de ruido (imnoise) debe ser un número entre 0.0 y 1.0.")
        return
    aplicar_y_mostrar_ruido(val)


def on_set_ruido(val):
    if not canal_rojo_puro:
        if entrada_ruido is not None and entrada_ruido.winfo_exists():
            entrada_ruido.delete(0, tk.END)
            entrada_ruido.insert(0, str(val))
        return
    aplicar_y_mostrar_ruido(val)


def abrir_imagen():
    global ancho_orig, alto_orig
    global canal_rojo_puro, canal_verde_puro, canal_azul_puro

    ruta = filedialog.askopenfilename(
        title="Seleccionar imagen",
        filetypes=[("Imágenes", "*.png *.jpg *.jpeg *.bmp *.ppm"), ("Todos los archivos", "*.*")]
    )
    if not ruta:
        return

    try:
        img_pil = Image.open(ruta).convert("RGB")
    except Exception as e:
        messagebox.showerror("Error", f"No se pudo abrir la imagen:\n{e}")
        return

    ancho_orig, alto_orig = img_pil.size
    pixeles = img_pil.load()

    filas_r, filas_g, filas_b = [], [], []

    for f in range(alto_orig):
        fr, fg, fb = bytearray(ancho_orig), bytearray(ancho_orig), bytearray(ancho_orig)
        for c in range(ancho_orig):
            r, g, b = pixeles[c, f][:3]
            fr[c], fg[c], fb[c] = r, g, b
        filas_r.append(fr)
        filas_g.append(fg)
        filas_b.append(fb)

    canal_rojo_puro = filas_r
    canal_verde_puro = filas_g
    canal_azul_puro = filas_b

    val_inicial = 0.05
    if entrada_ruido is not None and entrada_ruido.winfo_exists():
        try:
            val_inicial = float(entrada_ruido.get())
            val_inicial = max(0.0, min(1.0, val_inicial))
        except ValueError:
            val_inicial = 0.05

    densidad_pedida = simpledialog.askfloat(
        "Ruido Sal y Pimienta (imnoise)",
        "Ingrese el parámetro de ruido d (0.0 a 1.0):\n\n"
        "• 0.0 = Sin ruido (imagen original limpia)\n"
        "• 0.05 = Ruido estándar de prueba (5%)\n"
        "• 0.20 = Ruido moderado (20%)\n"
        "• 1.00 = Ruido total (100%)\n\n"
        "Densidad de ruido:",
        parent=ventana,
        minvalue=0.0,
        maxvalue=1.0,
        initialvalue=val_inicial
    )

    if densidad_pedida is None:
        densidad = val_inicial
    else:
        densidad = densidad_pedida

    aplicar_y_mostrar_ruido(densidad)


def mostrar_matriz_aplicada():
    if not nombre_filtro_aplicado_guardado:
        messagebox.showinfo("Información", "Primero procesa una imagen para ver los detalles.")
        return

    popup = tk.Toplevel(ventana)
    popup.title("Detalles de la Máscara W(x,y) Aplicada")
    popup.geometry("520x400")
    popup.configure(bg="#1a1a1a")
    popup.resizable(False, False)

    tk.Label(popup, text="Detalle de la Máscara Espacial W(x,y)", bg="#1a1a1a", fg="#4ec9b0", font=("Arial", 12, "bold")).pack(pady=(12, 4))
    tk.Label(popup, text=f"Técnica / Tipo: {nombre_filtro_aplicado_guardado}", bg="#1a1a1a", fg="#ffffff", font=("Arial", 10, "bold")).pack(pady=(0, 4))

    if mascara_aplicada_guardada:
        tam = len(mascara_aplicada_guardada)
        texto_sum = f"Dimensión: {tam}x{tam}  |  Suma de coeficientes: {suma_pesos_guardada:.4f}"
        if abs(suma_pesos_guardada) < 1e-6:
            texto_sum += "\n(Suma de pesos = 0: Filtro de acentuado pasa-altos; regiones constantes dan 0)"
        elif abs(suma_pesos_guardada - 1.0) > 1e-4:
            texto_sum += f"\n(Se normalizó dividiendo la convolución entre {suma_pesos_guardada:.2f})"
        else:
            texto_sum += "\n(La máscara está normalizada con suma total = 1.0)"

        tk.Label(popup, text=texto_sum, bg="#1a1a1a", fg="#9cdcfe", font=("Arial", 9), justify="center").pack(pady=(0, 8))

        frame_box = tk.Frame(popup, bg="#0d0d0d", padx=8, pady=8)
        frame_box.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 10))

        caja = tk.Text(frame_box, bg="#0d0d0d", fg="#dcdcaa", font=("Consolas", 10), wrap="none", bd=0)
        caja.pack(fill=tk.BOTH, expand=True)

        filas_str = []
        for fila in mascara_aplicada_guardada:
            vals = []
            for v in fila:
                if v == int(v):
                    vals.append(f"{int(v):>6}")
                else:
                    vals.append(f"{v:8.4f}")
            filas_str.append("  [ " + "  ".join(vals) + " ]")

        caja.insert(tk.END, "\n".join(filas_str))
        caja.config(state=tk.DISABLED)
    else:
        tam_ventana = len(entradas_matriz) if entradas_matriz else 3
        tk.Label(
            popup,
            text=f"\nNo posee una matriz de ponderaciones W(x,y),\nya que opera seleccionando directamente entre los píxeles\nde la ventana de vecindad de {tam_ventana}x{tam_ventana}.",
            bg="#1a1a1a", fg="#cccccc", font=("Arial", 10), justify="center"
        ).pack(pady=30)

    tk.Button(popup, text="Cerrar", command=popup.destroy, bg="#333333", fg="white", font=("Arial", 9, "bold"), width=12, relief=tk.FLAT).pack(pady=(0, 12))


def ejecutar_proceso():
    global vista_res_tk, mascara_aplicada_guardada, nombre_filtro_aplicado_guardado, suma_pesos_guardada
    global img_resultado_pil_nativa, zoom_factor, pan_x, pan_y

    if alto_orig == 0 or ancho_orig == 0:
        messagebox.showinfo("Aviso", "Primero carga una imagen.")
        return

    tam = len(entradas_matriz)
    if tam == 0:
        messagebox.showerror("Error", "Crea primero la cuadrícula de la máscara.")
        return

    tipo = combo_filtro.get()
    mascara = []

    if tipo == "Máscara (Convolución)":
        try:
            for i in range(tam):
                fila_vals = []
                for j in range(tam):
                    fila_vals.append(float(entradas_matriz[i][j].get()))
                mascara.append(fila_vals)
        except ValueError:
            messagebox.showerror("Error", "Verifica que todos los valores de la máscara sean números válidos.")
            return

        es_gauss_enteros = (
            tam == 3 and
            mascara[0] == [1.0, 2.0, 1.0] and
            mascara[1] == [2.0, 8.0, 2.0] and
            mascara[2] == [1.0, 2.0, 1.0]
        )
        es_acentuado_quemada = (
            tam == 3 and
            mascara[0] == [-1.0, -1.0, -1.0] and
            mascara[1] == [-1.0, 8.0, -1.0] and
            mascara[2] == [-1.0, -1.0, -1.0]
        )
        todos_iguales = all(val == mascara[0][0] for fila in mascara for val in fila)

        if es_gauss_enteros:
            nombre_filtro_aplicado_guardado = "Gaussiana Ejemplo"
        elif es_acentuado_quemada:
            nombre_filtro_aplicado_guardado = "Acentuado [-1, 8]"
        elif "Gaussiana" in nombre_mascara_activa:
            nombre_filtro_aplicado_guardado = "Gaussiana"
        elif todos_iguales:
            nombre_filtro_aplicado_guardado = "Media"
        else:
            nombre_filtro_aplicado_guardado = nombre_mascara_activa

        mascara_aplicada_guardada = mascara
        suma_pesos_guardada = sum(sum(fila) for fila in mascara)
    else:
        nombre_filtro_aplicado_guardado = tipo
        mascara_aplicada_guardada = []
        suma_pesos_guardada = 0.0

    try:
        salto_f = int(entrada_salto_f.get())
        salto_c = int(entrada_salto_c.get())
        if salto_f < 1 or salto_c < 1:
            raise ValueError
    except ValueError:
        messagebox.showerror("Error", "Los saltos deben ser números enteros mayores o iguales a 1.")
        return

    usar_pad = var_padding.get()
    borde = (tam - 1) // 2

    r_res, n_filas, n_cols = procesar_canal(canal_rojo, alto_orig, ancho_orig, mascara, tam, tam, tipo, salto_f, salto_c, usar_pad)
    g_res, _, _ = procesar_canal(canal_verde, alto_orig, ancho_orig, mascara, tam, tam, tipo, salto_f, salto_c, usar_pad)
    b_res, _, _ = procesar_canal(canal_azul, alto_orig, ancho_orig, mascara, tam, tam, tipo, salto_f, salto_c, usar_pad)

    buffer_res = bytearray(n_filas * n_cols * 3)
    pos = 0
    for f in range(n_filas):
        rf, gf, bf = r_res[f], g_res[f], b_res[f]
        for c in range(n_cols):
            buffer_res[pos] = rf[c]
            buffer_res[pos + 1] = gf[c]
            buffer_res[pos + 2] = bf[c]
            pos += 3
    img_resultado_pil_nativa = Image.frombytes("RGB", (n_cols, n_filas), bytes(buffer_res))

    zoom_factor = 1.0
    pan_x = 0.0
    pan_y = 0.0
    actualizar_render_zoom()

    pad_info = f"Sí (+{borde} px)" if usar_pad else "No"
    lbl_info_resultado_mascara.config(text=f"Máscara: {nombre_filtro_aplicado_guardado}")
    lbl_info_resultado_detalles.config(text=f"Salto: {salto_f}x{salto_c} | Pad: {pad_info} | Ventana: {tam}x{tam} | Rueda=Zoom Drag=Mover")
    btn_ver_mascara.config(state=tk.NORMAL)


def crear_interfaz():
    global ventana, lbl_tipo_mascara_activa, frame_cuadricula
    global entrada_tamano, combo_filtro, entrada_salto_f, entrada_salto_c, var_padding
    global lbl_img_original, lbl_info_original_dim
    global lbl_img_resultado, lbl_info_resultado_dim, lbl_info_resultado_mascara, lbl_info_resultado_detalles, btn_ver_mascara
    global canvas_resultado, lbl_zoom_val, entrada_ruido

    global lbl_frec_info_dim, entrada_ruido_frec, combo_tipo_pasa, combo_modelo_filtro
    global scale_d0, entry_d0, combo_modo_espectro
    global var_frec_quemada, var_modo_acentuado, scale_factor_a, lbl_tit_frec_filtrada
    global lbl_info_frec_orig, lbl_disp_frec_orig
    global lbl_info_frec_ruido, lbl_disp_frec_ruido
    global lbl_info_frec_fft, canvas_disp_frec_fft
    global lbl_info_frec_filtrada, lbl_disp_frec_filtrada
    global lbl_insp_uv, lbl_insp_radio, lbl_insp_real, lbl_insp_imag, lbl_insp_mag, lbl_insp_fase

    ventana = tk.Tk()
    ventana.title("PDI - Convolución Espacial, Reducción Dimensional y Filtrado en Frecuencia (FFT 2D)")
    ventana.geometry("1360x780")
    ventana.minsize(1260, 720)
    ventana.configure(bg="#181818")

    estilo = ttk.Style()
    try:
        estilo.theme_use("clam")
    except Exception:
        pass
    estilo.configure("TNotebook", background="#181818", borderwidth=0)
    estilo.configure(
        "TNotebook.Tab",
        background="#252526",
        foreground="#cccccc",
        padding=[20, 7],
        font=("Arial", 9, "bold")
    )
    estilo.map(
        "TNotebook.Tab",
        background=[("selected", "#0078d4"), ("active", "#333333")],
        foreground=[("selected", "#ffffff"), ("active", "#ffffff")]
    )

    notebook = ttk.Notebook(ventana)
    notebook.pack(fill=tk.BOTH, expand=True)

    tab_espacial = tk.Frame(notebook, bg="#181818")
    tab_frecuencia = tk.Frame(notebook, bg="#181818")

    notebook.add(tab_espacial, text="  1. Reducción y Convolución Espacial  ")
    notebook.add(tab_frecuencia, text="  2. Filtrado en Frecuencia (FFT 2D)  ")

                                                                               
    panel_izq = tk.Frame(tab_espacial, bg="#242424", padx=14, pady=12, width=320)
    panel_izq.pack(side=tk.LEFT, fill=tk.Y)

    frame_carga = tk.Frame(panel_izq, bg="#242424")
    frame_carga.pack(fill=tk.X, pady=(0, 6))

    tk.Button(
        frame_carga, text="1. Cargar Imagen", command=abrir_imagen,
        bg="#0078d4", fg="white", font=("Arial", 10, "bold"), width=25, relief=tk.FLAT
    ).pack(pady=(2, 4))

    frame_ruido = tk.Frame(frame_carga, bg="#242424")
    frame_ruido.pack(fill=tk.X, pady=(2, 2))

    tk.Label(frame_ruido, text="Ruido (0-1):", bg="#242424", fg="#ffd700", font=("Arial", 8, "bold")).pack(side=tk.LEFT)
    entrada_ruido = tk.Entry(frame_ruido, width=6, justify="center", font=("Arial", 8))
    entrada_ruido.insert(0, "0.05")
    entrada_ruido.pack(side=tk.LEFT, padx=3)

    tk.Button(
        frame_ruido, text="Aplicar", command=on_click_aplicar_ruido,
        bg="#3e3e42", fg="white", font=("Arial", 8, "bold"), relief=tk.FLAT, padx=4
    ).pack(side=tk.LEFT, padx=2)

    tk.Button(
        frame_ruido, text="0 (Limpia)", command=lambda: on_set_ruido(0.0),
        bg="#333333", fg="#aaaaaa", font=("Arial", 8), relief=tk.FLAT, padx=4
    ).pack(side=tk.LEFT, padx=2)

    tk.Label(panel_izq, text="2. Tamaño de la Máscara:", bg="#242424", fg="white", font=("Arial", 9, "bold")).pack(anchor="w", pady=(10, 2))
    frame_dim = tk.Frame(panel_izq, bg="#242424")
    frame_dim.pack(anchor="w")

    entrada_tamano = tk.Entry(frame_dim, width=4, justify="center")
    entrada_tamano.insert(0, "3")
    entrada_tamano.pack(side=tk.LEFT, padx=(0, 6))
    tk.Label(frame_dim, text="x N (ej. 3, 5)", bg="#242424", fg="gray", font=("Arial", 8)).pack(side=tk.LEFT, padx=(0, 8))
    tk.Button(frame_dim, text="Crear Cuadrícula", command=crear_cuadricula_vacia, bg="#3e3e42", fg="white", font=("Arial", 8, "bold"), relief=tk.FLAT).pack(side=tk.LEFT)

    tk.Label(panel_izq, text="3. Técnica de Filtro:", bg="#242424", fg="white", font=("Arial", 9, "bold")).pack(anchor="w", pady=(10, 2))
    combo_filtro = tk.StringVar(value="Máscara (Convolución)")
    combo_filtro.trace_add("write", on_cambio_filtro)

    menu_filtro = tk.OptionMenu(panel_izq, combo_filtro, "Máscara (Convolución)", "Mediana", "Moda")
    menu_filtro.config(bg="#333333", fg="white", font=("Arial", 9), highlightthickness=0, width=22)
    menu_filtro.pack(anchor="w")

    tk.Label(panel_izq, text="4. Máscara W(x,y) (Manual o Calculada):", bg="#242424", fg="#4ec9b0", font=("Arial", 9, "bold")).pack(anchor="w", pady=(10, 2))
    lbl_tipo_mascara_activa = tk.Label(panel_izq, text="Máscara Activa: Gaussiana Ejemplo", bg="#242424", fg="#ffd700", font=("Arial", 8, "italic"))
    lbl_tipo_mascara_activa.pack(anchor="w", pady=(0, 4))

    frame_cuadricula = tk.Frame(panel_izq, bg="#111111", padx=4, pady=4)
    frame_cuadricula.pack()

    frame_presets = tk.Frame(panel_izq, bg="#242424")
    frame_presets.pack(pady=4)
    tk.Button(frame_presets, text="Gaussiana", command=calcular_gaussiana_en_cuadricula, bg="#0e639c", fg="white", font=("Arial", 8, "bold"), relief=tk.FLAT).pack(side=tk.LEFT, padx=1)
    tk.Button(frame_presets, text="Ejemplo 3x3", command=cargar_preset_gaussiana_clase, bg="#333333", fg="white", font=("Arial", 8), relief=tk.FLAT).pack(side=tk.LEFT, padx=1)
    tk.Button(frame_presets, text="Media", command=cargar_preset_media, bg="#333333", fg="white", font=("Arial", 8), relief=tk.FLAT).pack(side=tk.LEFT, padx=1)
    tk.Button(frame_presets, text="[-1, 8]", command=cargar_preset_acentuado, bg="#8e44ad", fg="white", font=("Arial", 8, "bold"), relief=tk.FLAT).pack(side=tk.LEFT, padx=1)

    tk.Label(panel_izq, text="5. Salto de Reducción:", bg="#242424", fg="white", font=("Arial", 9, "bold")).pack(anchor="w", pady=(10, 2))
    frame_saltos = tk.Frame(panel_izq, bg="#242424")
    frame_saltos.pack(anchor="w")
    tk.Label(frame_saltos, text="Salto Filas (Y):", bg="#242424", fg="lightgray").grid(row=0, column=0, sticky="w")
    entrada_salto_f = tk.Entry(frame_saltos, width=5, justify="center")
    entrada_salto_f.insert(0, "2")
    entrada_salto_f.grid(row=0, column=1, padx=4, pady=2)

    tk.Label(frame_saltos, text="Salto Cols (X):", bg="#242424", fg="lightgray").grid(row=1, column=0, sticky="w")
    entrada_salto_c = tk.Entry(frame_saltos, width=5, justify="center")
    entrada_salto_c.insert(0, "2")
    entrada_salto_c.grid(row=1, column=1, padx=4, pady=2)

    var_padding = tk.BooleanVar(value=True)
    chk_pad = tk.Checkbutton(
        panel_izq, text="6. Aplicar Padding de ceros", variable=var_padding,
        bg="#242424", fg="white", selectcolor="#111111", font=("Arial", 9, "bold")
    )
    chk_pad.pack(anchor="w", pady=(10, 4))

    tk.Button(
        panel_izq, text="Procesar y Reducir", command=ejecutar_proceso,
        bg="#107c41", fg="white", font=("Arial", 10, "bold"), width=25, relief=tk.FLAT
    ).pack(pady=14)

    panel_der = tk.Frame(tab_espacial, bg="#181818", padx=12, pady=12)
    panel_der.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
    panel_der.columnconfigure(0, weight=1)
    panel_der.columnconfigure(1, weight=1)
    panel_der.rowconfigure(0, weight=1)

    col_izq = tk.Frame(panel_der, bg="#202020", padx=10, pady=10)
    col_izq.grid(row=0, column=0, sticky="nsew", padx=6, pady=4)

    tk.Label(col_izq, text="Imagen Original f(x,y)", bg="#202020", fg="white", font=("Arial", 11, "bold")).pack()
    lbl_info_original_dim = tk.Label(col_izq, text="Esperando imagen...", bg="#202020", fg="#888888", font=("Arial", 9))
    lbl_info_original_dim.pack(pady=(0, 2))

    tk.Label(col_izq, text="Resolución nativa", bg="#202020", fg="#666666", font=("Arial", 8, "italic")).pack(pady=(0, 2))
    lbl_espacio_simetria = tk.Label(col_izq, text="Vista previa de referencia original", bg="#202020", fg="#555555", font=("Arial", 8))
    lbl_espacio_simetria.pack(pady=(0, 2))

    col_der = tk.Frame(panel_der, bg="#202020", padx=10, pady=10)
    col_der.grid(row=0, column=1, sticky="nsew", padx=6, pady=4)

    tk.Label(col_der, text="Imagen Reducida g(x,y)", bg="#202020", fg="#4ec9b0", font=("Arial", 11, "bold")).pack()
    lbl_info_resultado_dim = tk.Label(col_der, text="Sin procesar", bg="#202020", fg="#4ec9b0", font=("Arial", 9, "bold"))
    lbl_info_resultado_dim.pack()

    lbl_info_resultado_mascara = tk.Label(col_der, text="Máscara: Ninguna procesada", bg="#202020", fg="#ffd700", font=("Arial", 9, "bold"))
    lbl_info_resultado_mascara.pack(pady=(1, 1))

    lbl_info_resultado_detalles = tk.Label(col_der, text="Ajusta los parámetros y presiona Procesar", bg="#202020", fg="#888888", font=("Arial", 8))
    lbl_info_resultado_detalles.pack(pady=(0, 2))

    btn_ver_mascara = tk.Button(
        col_der, text="Ver Matriz W(x,y) Aplicada", command=mostrar_matriz_aplicada,
        bg="#2d2d30", fg="#4ec9b0", font=("Arial", 8, "bold"), relief=tk.FLAT, cursor="hand2", state=tk.DISABLED
    )
    btn_ver_mascara.pack(pady=(0, 2))

    frame_zoom_bar = tk.Frame(col_der, bg="#202020")
    frame_zoom_bar.pack(pady=(2, 4))

    tk.Button(
        frame_zoom_bar, text="🔍➖ Alejar", command=lambda: aplicar_delta_zoom(1.0 / 1.25),
        bg="#2d2d30", fg="white", font=("Arial", 8, "bold"), relief=tk.FLAT, padx=6, cursor="hand2"
    ).pack(side=tk.LEFT, padx=2)

    lbl_zoom_val = tk.Label(frame_zoom_bar, text="Zoom: 1.00×", bg="#202020", fg="#4ec9b0", font=("Arial", 9, "bold"), width=12)
    lbl_zoom_val.pack(side=tk.LEFT, padx=2)

    tk.Button(
        frame_zoom_bar, text="🔍➕ Acercar", command=lambda: aplicar_delta_zoom(1.25),
        bg="#2d2d30", fg="white", font=("Arial", 8, "bold"), relief=tk.FLAT, padx=6, cursor="hand2"
    ).pack(side=tk.LEFT, padx=2)

    tk.Button(
        frame_zoom_bar, text="↺ 1.0× Reset", command=resetear_zoom,
        bg="#2d2d30", fg="#ffd700", font=("Arial", 8, "bold"), relief=tk.FLAT, padx=6, cursor="hand2"
    ).pack(side=tk.LEFT, padx=2)

    tk.Frame(col_izq, bg="#202020", height=28).pack(pady=(0, 4))

    frame_display_orig = tk.Frame(col_izq, bg="#0d0d0d", width=MAX_ANCHO_PREVIEW, height=MAX_ALTO_PREVIEW)
    frame_display_orig.pack(expand=True)
    frame_display_orig.pack_propagate(False)
    lbl_img_original = tk.Label(frame_display_orig, bg="#0d0d0d", text="Sin imagen cargada", fg="#555555")
    lbl_img_original.pack(expand=True)

    frame_display_res = tk.Frame(col_der, bg="#0d0d0d", width=MAX_ANCHO_PREVIEW, height=MAX_ALTO_PREVIEW)
    frame_display_res.pack(expand=True)
    frame_display_res.pack_propagate(False)

    canvas_resultado = tk.Canvas(frame_display_res, bg="#0d0d0d", width=MAX_ANCHO_PREVIEW, height=MAX_ALTO_PREVIEW, highlightthickness=0)
    canvas_resultado.pack(expand=True)
    canvas_resultado.create_text(
        MAX_ANCHO_PREVIEW // 2, MAX_ALTO_PREVIEW // 2,
        text="Sin resultado\n\nAjusta los parámetros y presiona 'Procesar y Reducir'",
        fill="#555555", font=("Arial", 10), justify="center", tags="placeholder"
    )

    canvas_resultado.bind("<MouseWheel>", on_scroll_zoom)
    canvas_resultado.bind("<Button-4>", lambda e: aplicar_delta_zoom(1.25, e.x, e.y))
    canvas_resultado.bind("<Button-5>", lambda e: aplicar_delta_zoom(1.0 / 1.25, e.x, e.y))
    canvas_resultado.bind("<ButtonPress-1>", on_drag_inicio)
    canvas_resultado.bind("<B1-Motion>", on_drag_mover)
    canvas_resultado.bind("<Double-Button-1>", resetear_zoom)

    lbl_img_resultado = canvas_resultado

    cargar_preset_gaussiana_clase()

                                                                               
    container_izq_frec = tk.Frame(tab_frecuencia, bg="#242424", width=335)
    container_izq_frec.pack(side=tk.LEFT, fill=tk.Y)
    container_izq_frec.pack_propagate(False)

    canvas_scroll_frec = tk.Canvas(container_izq_frec, bg="#242424", highlightthickness=0)
    scrollbar_frec = ttk.Scrollbar(container_izq_frec, orient="vertical", command=canvas_scroll_frec.yview)

    panel_izq_frec = tk.Frame(canvas_scroll_frec, bg="#242424", padx=8, pady=4)
    win_frec_id = canvas_scroll_frec.create_window((0, 0), window=panel_izq_frec, anchor="nw")

    def _on_panel_config(event=None):
        canvas_scroll_frec.update_idletasks()
        canvas_scroll_frec.configure(scrollregion=canvas_scroll_frec.bbox("all"))

    panel_izq_frec.bind("<Configure>", _on_panel_config)

    def _on_canvas_config(event):
        canvas_scroll_frec.itemconfig(win_frec_id, width=event.width)

    canvas_scroll_frec.bind("<Configure>", _on_canvas_config)
    canvas_scroll_frec.configure(yscrollcommand=scrollbar_frec.set)

    scrollbar_frec.pack(side=tk.RIGHT, fill=tk.Y)
    canvas_scroll_frec.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    def _on_mousewheel_panel(event):
        if event.delta:
            canvas_scroll_frec.yview_scroll(int(-1 * (event.delta / 120)), "units")
        elif event.num == 4:
            canvas_scroll_frec.yview_scroll(-1, "units")
        elif event.num == 5:
            canvas_scroll_frec.yview_scroll(1, "units")

    def _bind_mousewheel(event=None):
        canvas_scroll_frec.bind_all("<MouseWheel>", _on_mousewheel_panel)
        canvas_scroll_frec.bind_all("<Button-4>", _on_mousewheel_panel)
        canvas_scroll_frec.bind_all("<Button-5>", _on_mousewheel_panel)

    def _unbind_mousewheel(event=None):
        canvas_scroll_frec.unbind_all("<MouseWheel>")
        canvas_scroll_frec.unbind_all("<Button-4>")
        canvas_scroll_frec.unbind_all("<Button-5>")

    container_izq_frec.bind("<Enter>", _bind_mousewheel)
    container_izq_frec.bind("<Leave>", _unbind_mousewheel)

                              
    frame_carga_frec = tk.Frame(panel_izq_frec, bg="#242424")
    frame_carga_frec.pack(fill=tk.X, pady=(0, 4))

    tk.Button(
        frame_carga_frec, text="1. Cargar Imagen", command=abrir_imagen,
        bg="#0078d4", fg="white", font=("Arial", 10, "bold"), width=25, relief=tk.FLAT
    ).pack(pady=(2, 3))

    frame_ruido_frec = tk.Frame(frame_carga_frec, bg="#242424")
    frame_ruido_frec.pack(fill=tk.X, pady=(2, 2))

    tk.Label(frame_ruido_frec, text="Ruido (0-1):", bg="#242424", fg="#ffd700", font=("Arial", 8, "bold")).pack(side=tk.LEFT)
    entrada_ruido_frec = tk.Entry(frame_ruido_frec, width=6, justify="center", font=("Arial", 8))
    entrada_ruido_frec.insert(0, "0.05")
    entrada_ruido_frec.pack(side=tk.LEFT, padx=3)

    tk.Button(
        frame_ruido_frec, text="Aplicar", command=on_click_aplicar_ruido_frec,
        bg="#3e3e42", fg="white", font=("Arial", 8, "bold"), relief=tk.FLAT, padx=4
    ).pack(side=tk.LEFT, padx=2)

    tk.Button(
        frame_ruido_frec, text="0 (Limpia)", command=lambda: on_set_ruido(0.0),
        bg="#333333", fg="#aaaaaa", font=("Arial", 8), relief=tk.FLAT, padx=4
    ).pack(side=tk.LEFT, padx=2)

    lbl_frec_info_dim = tk.Label(frame_carga_frec, text="Esperando imagen...", bg="#242424", fg="#888888", font=("Arial", 8))
    lbl_frec_info_dim.pack(pady=(2, 0))

                                     
    tk.Label(panel_izq_frec, text="2. Tipo de Filtro en Frecuencia:", bg="#242424", fg="white", font=("Arial", 9, "bold")).pack(anchor="w", pady=(6, 2))
    combo_tipo_pasa = tk.StringVar(value="Pasa Bajos (Low-Pass)")
    menu_tipo_pasa = tk.OptionMenu(
        panel_izq_frec, combo_tipo_pasa,
        "Pasa Bajos (Low-Pass)", "Pasa Altos (High-Pass)",
        command=lambda _: ejecutar_filtro_frecuencia()
    )
    menu_tipo_pasa.config(bg="#333333", fg="white", font=("Arial", 9), highlightthickness=0, width=22)
                                          
    tk.Label(panel_izq_frec, text="3. Frecuencia de Corte D₀ (Radio px):", bg="#242424", fg="white", font=("Arial", 9, "bold")).pack(anchor="w", pady=(6, 2))
    frame_d0 = tk.Frame(panel_izq_frec, bg="#242424")
    frame_d0.pack(fill=tk.X, pady=(0, 2))

    scale_d0 = tk.Scale(
        frame_d0, from_=2, to=200, orient=tk.HORIZONTAL, bg="#242424", fg="white",
        highlightthickness=0, troughcolor="#111111", activebackground="#0078d4",
        command=on_scale_d0_change
    )
    scale_d0.set(40)
    scale_d0.pack(side=tk.LEFT, fill=tk.X, expand=True)
    scale_d0.bind("<ButtonRelease-1>", lambda _: ejecutar_filtro_frecuencia())

    entry_d0 = tk.Entry(frame_d0, width=5, justify="center", font=("Arial", 9))
    entry_d0.insert(0, "40")
    entry_d0.pack(side=tk.LEFT, padx=(6, 0))
    entry_d0.bind("<Return>", lambda e: [on_entry_d0_change(), ejecutar_filtro_frecuencia()])

    tk.Button(
        panel_izq_frec, text="⚡ Aplicar Filtro FFT", command=ejecutar_filtro_frecuencia,
        bg="#107c41", fg="white", font=("Arial", 10, "bold"), width=25, relief=tk.FLAT
    ).pack(pady=(6, 6))

                                 
    tk.Label(panel_izq_frec, text="4. Inspector FFT F(u, v) Complejo:", bg="#242424", fg="#4ec9b0", font=("Arial", 9, "bold")).pack(anchor="w", pady=(3, 2))

    frame_insp = tk.Frame(panel_izq_frec, bg="#111111", padx=8, pady=5, bd=1, relief=tk.SOLID)
    frame_insp.pack(fill=tk.X, pady=(0, 6))

    tk.Label(frame_insp, text="🔬 Coeficiente F(u,v) = R + j·I", bg="#111111", fg="#ffd700", font=("Arial", 8, "bold")).pack(anchor="w")
    tk.Label(frame_insp, text="(Mueve el cursor sobre el espectro FFT)", bg="#111111", fg="#888888", font=("Arial", 7, "italic")).pack(anchor="w", pady=(0, 2))

    lbl_insp_uv = tk.Label(frame_insp, text="Frecuencia (u, v): (0, 0)  [Centro DC]", bg="#111111", fg="#ffffff", font=("Consolas", 8, "bold"))
    lbl_insp_uv.pack(anchor="w")

    lbl_insp_radio = tk.Label(frame_insp, text="Radio D(u, v): 0.00 px", bg="#111111", fg="#cccccc", font=("Consolas", 8))
    lbl_insp_radio.pack(anchor="w")

    lbl_insp_real = tk.Label(frame_insp, text="Parte Real: +0.000e+00", bg="#111111", fg="#4ec9b0", font=("Consolas", 8))
    lbl_insp_real.pack(anchor="w")

    lbl_insp_imag = tk.Label(frame_insp, text="Parte Imag: +0.000e+00 j", bg="#111111", fg="#ce9178", font=("Consolas", 8, "bold"))
    lbl_insp_imag.pack(anchor="w")

    lbl_insp_mag = tk.Label(frame_insp, text="Magnitud |F|: 0.000e+00", bg="#111111", fg="#9cdcfe", font=("Consolas", 8))
    lbl_insp_mag.pack(anchor="w")

    lbl_insp_fase = tk.Label(frame_insp, text="Fase φ: +0.0°", bg="#111111", fg="#dcdcaa", font=("Consolas", 8))
    lbl_insp_fase.pack(anchor="w")

                                           
    tk.Label(panel_izq_frec, text="5. Modo Visual del Espectro:", bg="#242424", fg="white", font=("Arial", 8, "bold")).pack(anchor="w", pady=(2, 1))
    combo_modo_espectro = tk.StringVar(value="Magnitud Logarítmica log(1+|F|)")
    combo_modo_espectro.trace_add("write", on_cambio_modo_espectro)
    menu_modo = tk.OptionMenu(
        panel_izq_frec, combo_modo_espectro,
        "Magnitud Logarítmica log(1+|F|)",
        "Parte Real Re{F(u,v)}",
        "Parte Imaginaria Im{F(u,v)}",
        "Ángulo de Fase φ(u,v)",
        "Máscara Filtro H(u,v)"
    )
    menu_modo.config(bg="#333333", fg="white", font=("Arial", 8), highlightthickness=0, width=24)
    menu_modo.pack(anchor="w")

                                       
    frame_post_frec = tk.LabelFrame(
        panel_izq_frec,
        text="6. Filtro de Acentuado sobre IFFT:",
        bg="#242424", fg="#ffd700", font=("Arial", 8, "bold"),
        padx=6, pady=5
    )
    frame_post_frec.pack(fill=tk.X, pady=(6, 2))

    var_frec_quemada = tk.BooleanVar(value=False)
    chk_frec_quemada = tk.Checkbutton(
        frame_post_frec,
        text="Aplicar Filtro de Acentuado",
        variable=var_frec_quemada,
        command=actualizar_vista_cuadrante_4,
        bg="#242424", fg="white", selectcolor="#111111", font=("Arial", 8, "bold")
    )
    chk_frec_quemada.pack(anchor="w", pady=(0, 2))

    var_modo_acentuado = tk.StringVar(value="enfasis")
    frame_radios_modo = tk.Frame(frame_post_frec, bg="#242424")
    frame_radios_modo.pack(fill=tk.X, pady=(1, 2))

    rb_enfasis = tk.Radiobutton(
        frame_radios_modo, text="Énfasis (IFFT + PA)", variable=var_modo_acentuado,
        value="enfasis", command=actualizar_vista_cuadrante_4,
        bg="#242424", fg="#4ec9b0", selectcolor="#111111", font=("Arial", 8, "bold")
    )
    rb_enfasis.pack(anchor="w")

    rb_laplaciano = tk.Radiobutton(
        frame_radios_modo, text="Solo Bordes (PA puro)", variable=var_modo_acentuado,
        value="laplaciano", command=actualizar_vista_cuadrante_4,
        bg="#242424", fg="#cccccc", selectcolor="#111111", font=("Arial", 8)
    )
    rb_laplaciano.pack(anchor="w")

    frame_slider_a = tk.Frame(frame_post_frec, bg="#242424")
    frame_slider_a.pack(fill=tk.X, pady=(1, 2))
    tk.Label(frame_slider_a, text="Ganancia A:", bg="#242424", fg="lightgray", font=("Arial", 8)).pack(side=tk.LEFT)
    scale_factor_a = tk.Scale(
        frame_slider_a, from_=0.2, to=2.0, resolution=0.1, orient=tk.HORIZONTAL,
        bg="#242424", fg="white", highlightthickness=0, troughcolor="#111111",
        activebackground="#0078d4", length=110
    )
    scale_factor_a.set(1.0)
    scale_factor_a.pack(side=tk.RIGHT, fill=tk.X, expand=True)
    scale_factor_a.bind("<ButtonRelease-1>", lambda _: actualizar_vista_cuadrante_4())

    tk.Button(
        frame_post_frec, text="📄 Ver Matriz / Fórmula 9A-1",
        command=mostrar_matriz_quemada_popup,
        bg="#2d2d30", fg="#4ec9b0", font=("Arial", 8, "bold"), relief=tk.FLAT
    ).pack(fill=tk.X, pady=(3, 0))

                                                 
    panel_der_frec = tk.Frame(tab_frecuencia, bg="#181818", padx=8, pady=6)
    panel_der_frec.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
    panel_der_frec.columnconfigure(0, weight=1)
    panel_der_frec.columnconfigure(1, weight=1)
    panel_der_frec.rowconfigure(0, weight=1)
    panel_der_frec.rowconfigure(1, weight=1)

                                                 
    c1 = tk.Frame(panel_der_frec, bg="#202020", padx=6, pady=4, bd=1, relief=tk.SOLID)
    c1.grid(row=0, column=0, padx=4, pady=4, sticky="nsew")
    tk.Label(c1, text="1. Imagen Original f(x,y)", bg="#202020", fg="white", font=("Arial", 10, "bold")).pack()
    lbl_info_frec_orig = tk.Label(c1, text="Esperando imagen...", bg="#202020", fg="#888888", font=("Arial", 8))
    lbl_info_frec_orig.pack(pady=(0, 2))
    frame_disp_c1 = tk.Frame(c1, bg="#0d0d0d", width=MAX_W_QUAD, height=MAX_H_QUAD)
    frame_disp_c1.pack(expand=True, fill=tk.BOTH)
    frame_disp_c1.pack_propagate(False)
    lbl_disp_frec_orig = tk.Label(frame_disp_c1, bg="#0d0d0d", text="Sin imagen cargada", fg="#555555")
    lbl_disp_frec_orig.pack(expand=True)

                                                  
    c2 = tk.Frame(panel_der_frec, bg="#202020", padx=6, pady=4, bd=1, relief=tk.SOLID)
    c2.grid(row=0, column=1, padx=4, pady=4, sticky="nsew")
    tk.Label(c2, text="2. Imagen con Ruido I(x,y)", bg="#202020", fg="#ffd700", font=("Arial", 10, "bold")).pack()
    lbl_info_frec_ruido = tk.Label(c2, text="Sin ruido", bg="#202020", fg="#ffd700", font=("Arial", 8))
    lbl_info_frec_ruido.pack(pady=(0, 2))
    frame_disp_c2 = tk.Frame(c2, bg="#0d0d0d", width=MAX_W_QUAD, height=MAX_H_QUAD)
    frame_disp_c2.pack(expand=True, fill=tk.BOTH)
    frame_disp_c2.pack_propagate(False)
    lbl_disp_frec_ruido = tk.Label(frame_disp_c2, bg="#0d0d0d", text="Sin imagen con ruido", fg="#555555")
    lbl_disp_frec_ruido.pack(expand=True)

                                                  
    c3 = tk.Frame(panel_der_frec, bg="#202020", padx=6, pady=4, bd=1, relief=tk.SOLID)
    c3.grid(row=1, column=0, padx=4, pady=4, sticky="nsew")
    tk.Label(c3, text="3. Espectro FFT 2D If(u,v)", bg="#202020", fg="#4ec9b0", font=("Arial", 10, "bold")).pack()
    lbl_info_frec_fft = tk.Label(c3, text="Mueve el ratón para inspeccionar frecuencia y componentes imaginarios", bg="#202020", fg="#4ec9b0", font=("Arial", 8))
    lbl_info_frec_fft.pack(pady=(0, 2))
    frame_disp_c3 = tk.Frame(c3, bg="#0d0d0d", width=MAX_W_QUAD, height=MAX_H_QUAD)
    frame_disp_c3.pack(expand=True, fill=tk.BOTH)
    frame_disp_c3.pack_propagate(False)
    canvas_disp_frec_fft = tk.Canvas(frame_disp_c3, bg="#0d0d0d", width=MAX_W_QUAD, height=MAX_H_QUAD, highlightthickness=0, cursor="crosshair")
    canvas_disp_frec_fft.pack(expand=True)
    canvas_disp_frec_fft.create_text(
        MAX_W_QUAD // 2, MAX_H_QUAD // 2,
        text="Sin espectro FFT\n(Carga una imagen para generar la transformada de Fourier 2D)",
        fill="#555555", font=("Arial", 9), justify="center", tags="placeholder"
    )
    canvas_disp_frec_fft.bind("<Motion>", on_hover_espectro_fft)
    canvas_disp_frec_fft.bind("<Leave>", on_leave_espectro_fft)

                                                                         
    c4 = tk.Frame(panel_der_frec, bg="#202020", padx=6, pady=4, bd=1, relief=tk.SOLID)
    c4.grid(row=1, column=1, padx=4, pady=4, sticky="nsew")
    lbl_tit_frec_filtrada = tk.Label(c4, text="4. Imagen Filtrada en Frecuencia (IFFT)", bg="#202020", fg="#9cdcfe", font=("Arial", 10, "bold"))
    lbl_tit_frec_filtrada.pack()
    lbl_info_frec_filtrada = tk.Label(c4, text="Sin filtrar (Presiona 'Aplicar Filtro FFT')", bg="#202020", fg="#9cdcfe", font=("Arial", 8))
    lbl_info_frec_filtrada.pack(pady=(0, 2))
    frame_disp_c4 = tk.Frame(c4, bg="#0d0d0d", width=MAX_W_QUAD, height=MAX_H_QUAD)
    frame_disp_c4.pack(expand=True, fill=tk.BOTH)
    frame_disp_c4.pack_propagate(False)
    lbl_disp_frec_filtrada = tk.Label(frame_disp_c4, bg="#0d0d0d", text="Sin resultado filtrado", fg="#555555")
    lbl_disp_frec_filtrada.pack(expand=True)

    return ventana


def iniciar_interfaz():
    app = crear_interfaz()
    app.mainloop()


if __name__ == "__main__":
    iniciar_interfaz()
