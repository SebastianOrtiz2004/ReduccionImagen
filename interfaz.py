import tkinter as tk
from tkinter import filedialog, messagebox
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
)

# Variables globales para almacenar matrices originales en uint8
ancho_orig = 0
alto_orig = 0
canal_rojo = []
canal_verde = []
canal_azul = []

vista_orig_tk = None
vista_res_tk = None
entradas_matriz = []

# Variables globales para rastrear la máscara aplicada y sus pesos
nombre_mascara_activa = "Gaussiana Ejemplo"
mascara_aplicada_guardada = []
nombre_filtro_aplicado_guardado = ""
suma_pesos_guardada = 1.0

# Variables globales para el zoom interactivo de la imagen reducida
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

# Referencias a widgets dinámicos de interfaz
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

# Dimensiones máximas asignadas para la vista previa simétrica
MAX_ANCHO_PREVIEW = 440
MAX_ALTO_PREVIEW = 460


# ------------------------------------------------------------------------------
# VISUALIZACIÓN SIMÉTRICA (SIN MODIFICAR LOS DATOS DE MEMORIA)
# ------------------------------------------------------------------------------
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


# ------------------------------------------------------------------------------
# ZOOM INTERACTIVO Y NAVEGACIÓN EN LA IMAGEN RESULTANTE
# ------------------------------------------------------------------------------
def actualizar_render_zoom():
    global vista_res_tk, img_canvas_id
    if img_resultado_pil_nativa is None or canvas_resultado is None:
        return

    n_cols, n_filas = img_resultado_pil_nativa.size

    # Escala base simétrica respecto a la imagen original cargada
    # para que la reducción física (tamaño en pantalla) sea real y evidente
    if ancho_orig > 0 and alto_orig > 0:
        escala_base = min(MAX_ANCHO_PREVIEW / ancho_orig, MAX_ALTO_PREVIEW / alto_orig, 1.0)
    else:
        escala_base = 1.0

    # Tamaño base en pantalla (exactamente al tamaño al que se redujo la imagen)
    base_w = max(1, int(n_cols * escala_base))
    base_h = max(1, int(n_filas * escala_base))

    # Tamaño aplicando el factor de zoom interactivo a partir del tamaño reducido
    target_w = max(1, int(base_w * zoom_factor))
    target_h = max(1, int(base_h * zoom_factor))

    # Prevenir desborde de memoria limitando dimensiones a 4000px
    target_w = min(target_w, 4000)
    target_h = min(target_h, 4000)

    # Reescalado de alta calidad con Lanczos
    img_render = img_resultado_pil_nativa.resize((target_w, target_h), Image.Resampling.LANCZOS)
    vista_res_tk = ImageTk.PhotoImage(img_render)

    canvas_resultado.delete("all")
    cx = (MAX_ANCHO_PREVIEW // 2) + pan_x
    cy = (MAX_ALTO_PREVIEW // 2) + pan_y
    img_canvas_id = canvas_resultado.create_image(cx, cy, anchor="center", image=vista_res_tk)
    canvas_resultado.config(cursor="fleur")

    # Actualizar texto del indicador de zoom
    if lbl_zoom_val is not None and lbl_zoom_val.winfo_exists():
        lbl_zoom_val.config(text=f"Zoom: {zoom_factor:.2f}×")

    # Actualizar dimensiones en el panel
    if lbl_info_resultado_dim is not None and lbl_info_resultado_dim.winfo_exists():
        if abs(zoom_factor - 1.0) < 0.01:
            lbl_info_resultado_dim.config(text=f"{n_filas} x {n_cols} px  (1.0× Tamaño Reducido)")
        else:
            lbl_info_resultado_dim.config(text=f"{n_filas} x {n_cols} px  |  🔍 Zoom {zoom_factor:.2f}×")


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


# ------------------------------------------------------------------------------
# GESTIÓN Y CÁLCULO DE LA CUADRÍCULA DE LA MÁSCARA
# ------------------------------------------------------------------------------
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

    # Escribimos los pesos en las casillas editables
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


def on_cambio_filtro(*args):
    tipo = combo_filtro.get()
    if tipo == "Mediana":
        actualizar_indicador_mascara("Mediana")
    elif tipo == "Moda":
        actualizar_indicador_mascara("Moda")
    else:
        actualizar_indicador_mascara()


# ------------------------------------------------------------------------------
# EVENTOS DE CARGA Y EJECUCIÓN
# ------------------------------------------------------------------------------
def abrir_imagen():
    global ancho_orig, alto_orig, canal_rojo, canal_verde, canal_azul, vista_orig_tk

    ruta = filedialog.askopenfilename(
        title="Seleccionar imagen",
        filetypes=[("Imágenes", "*.png *.jpg *.jpeg *.bmp *.ppm"), ("Todos los archivos", "*.*")]
    )
    if not ruta:
        return

    img_pil = Image.open(ruta).convert("RGB")
    ancho_orig, alto_orig = img_pil.size
    pixeles = img_pil.load()

    filas_r, filas_g, filas_b = [], [], []

    # Carga completa al 100% de píxeles reales en memoria RAM
    for f in range(alto_orig):
        fr, fg, fb = bytearray(ancho_orig), bytearray(ancho_orig), bytearray(ancho_orig)
        for c in range(ancho_orig):
            r, g, b = pixeles[c, f][:3]
            fr[c], fg[c], fb[c] = r, g, b
        filas_r.append(fr)
        filas_g.append(fg)
        filas_b.append(fb)

    canal_rojo = filas_r
    canal_verde = filas_g
    canal_azul = filas_b

    vista_orig_tk = generar_foto_preview(canal_rojo, canal_verde, canal_azul, alto_orig, ancho_orig)
    lbl_img_original.config(image=vista_orig_tk, text="")
    lbl_info_original_dim.config(text=f"{alto_orig} x {ancho_orig} px")

    # Limpiar resultado anterior y reiniciar estado de zoom
    global img_resultado_pil_nativa, zoom_factor, pan_x, pan_y
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
        if abs(suma_pesos_guardada - 1.0) > 1e-4:
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

    # Solo leemos los números de la matriz si es convolución lineal
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

        # Identificar la máscara con precisión
        es_gauss_enteros = (tam == 3 and
                            mascara[0] == [1.0, 2.0, 1.0] and
                            mascara[1] == [2.0, 8.0, 2.0] and
                            mascara[2] == [1.0, 2.0, 1.0])
        todos_iguales = all(val == mascara[0][0] for fila in mascara for val in fila)

        if es_gauss_enteros:
            nombre_filtro_aplicado_guardado = "Gaussiana Ejemplo"
        elif "Gaussiana" in nombre_mascara_activa:
            nombre_filtro_aplicado_guardado = "Gaussiana"
        elif todos_iguales:
            nombre_filtro_aplicado_guardado = "Media"
        else:
            nombre_filtro_aplicado_guardado = nombre_mascara_activa

        mascara_aplicada_guardada = mascara
        suma_pesos_guardada = sum(sum(fila) for fila in mascara)
        if suma_pesos_guardada == 0:
            suma_pesos_guardada = 1.0
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

    # Procesamiento matemático a resolución nativa
    r_res, n_filas, n_cols = procesar_canal(canal_rojo, alto_orig, ancho_orig, mascara, tam, tam, tipo, salto_f, salto_c, usar_pad)
    g_res, _, _ = procesar_canal(canal_verde, alto_orig, ancho_orig, mascara, tam, tam, tipo, salto_f, salto_c, usar_pad)
    b_res, _, _ = procesar_canal(canal_azul, alto_orig, ancho_orig, mascara, tam, tam, tipo, salto_f, salto_c, usar_pad)

    # Construimos la imagen PIL nativa (100% de píxeles reducidos reales)
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

    # Reiniciar zoom al 1.0x (tamaño exacto reducido) y centrado
    zoom_factor = 1.0
    pan_x = 0.0
    pan_y = 0.0
    actualizar_render_zoom()

    pad_info = f"Sí (+{borde} px)" if usar_pad else "No"
    lbl_info_resultado_mascara.config(text=f"Máscara: {nombre_filtro_aplicado_guardado}")
    lbl_info_resultado_detalles.config(text=f"Salto: {salto_f}x{salto_c} | Pad: {pad_info} | Ventana: {tam}x{tam} | Rueda=Zoom Drag=Mover")
    btn_ver_mascara.config(state=tk.NORMAL)


# ------------------------------------------------------------------------------
# CONSTRUCCIÓN DE LA INTERFAZ GRÁFICA
# ------------------------------------------------------------------------------
def crear_interfaz():
    global ventana, lbl_tipo_mascara_activa, frame_cuadricula
    global entrada_tamano, combo_filtro, entrada_salto_f, entrada_salto_c, var_padding
    global lbl_img_original, lbl_info_original_dim
    global lbl_img_resultado, lbl_info_resultado_dim, lbl_info_resultado_mascara, lbl_info_resultado_detalles, btn_ver_mascara
    global canvas_resultado, lbl_zoom_val

    ventana = tk.Tk()
    ventana.title("PDI - Convolución Espacial y Reducción por Salto")
    ventana.geometry("1280x730")
    ventana.configure(bg="#181818")

    # Panel lateral izquierdo
    panel_izq = tk.Frame(ventana, bg="#242424", padx=14, pady=12, width=320)
    panel_izq.pack(side=tk.LEFT, fill=tk.Y)

    tk.Button(
        panel_izq, text="1. Cargar Imagen", command=abrir_imagen,
        bg="#0078d4", fg="white", font=("Arial", 10, "bold"), width=25, relief=tk.FLAT
    ).pack(pady=4)

    # 1. Tamaño de la máscara
    tk.Label(panel_izq, text="2. Tamaño de la Máscara:", bg="#242424", fg="white", font=("Arial", 9, "bold")).pack(anchor="w", pady=(10, 2))
    frame_dim = tk.Frame(panel_izq, bg="#242424")
    frame_dim.pack(anchor="w")

    entrada_tamano = tk.Entry(frame_dim, width=4, justify="center")
    entrada_tamano.insert(0, "3")
    entrada_tamano.pack(side=tk.LEFT, padx=(0, 6))
    tk.Label(frame_dim, text="x N (ej. 3, 5)", bg="#242424", fg="gray", font=("Arial", 8)).pack(side=tk.LEFT, padx=(0, 8))
    tk.Button(frame_dim, text="Crear Cuadrícula", command=crear_cuadricula_vacia, bg="#3e3e42", fg="white", font=("Arial", 8, "bold"), relief=tk.FLAT).pack(side=tk.LEFT)

    # 2. Selector de Filtro
    tk.Label(panel_izq, text="3. Técnica de Filtro:", bg="#242424", fg="white", font=("Arial", 9, "bold")).pack(anchor="w", pady=(10, 2))
    combo_filtro = tk.StringVar(value="Máscara (Convolución)")
    combo_filtro.trace_add("write", on_cambio_filtro)

    menu_filtro = tk.OptionMenu(panel_izq, combo_filtro, "Máscara (Convolución)", "Mediana", "Moda")
    menu_filtro.config(bg="#333333", fg="white", font=("Arial", 9), highlightthickness=0, width=22)
    menu_filtro.pack(anchor="w")

    # 3. Cuadrícula de la máscara y botones de cálculo
    tk.Label(panel_izq, text="4. Máscara W(x,y) (Manual o Calculada):", bg="#242424", fg="#4ec9b0", font=("Arial", 9, "bold")).pack(anchor="w", pady=(10, 2))
    lbl_tipo_mascara_activa = tk.Label(panel_izq, text="Máscara Activa: Gaussiana Ejemplo", bg="#242424", fg="#ffd700", font=("Arial", 8, "italic"))
    lbl_tipo_mascara_activa.pack(anchor="w", pady=(0, 4))

    frame_cuadricula = tk.Frame(panel_izq, bg="#111111", padx=4, pady=4)
    frame_cuadricula.pack()

    frame_presets = tk.Frame(panel_izq, bg="#242424")
    frame_presets.pack(pady=4)
    tk.Button(frame_presets, text="Calcular Gaussiana", command=calcular_gaussiana_en_cuadricula, bg="#0e639c", fg="white", font=("Arial", 8, "bold"), relief=tk.FLAT).pack(side=tk.LEFT, padx=2)
    tk.Button(frame_presets, text="Gaussiana Ejemplo", command=cargar_preset_gaussiana_clase, bg="#333333", fg="white", font=("Arial", 8), relief=tk.FLAT).pack(side=tk.LEFT, padx=2)
    tk.Button(frame_presets, text="Media", command=cargar_preset_media, bg="#333333", fg="white", font=("Arial", 8), relief=tk.FLAT).pack(side=tk.LEFT, padx=2)

    # 4. Saltos de reducción
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

    # 5. Padding
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

    # Panel derecho comparativo simétrico (50% y 50%)
    panel_der = tk.Frame(ventana, bg="#181818", padx=12, pady=12)
    panel_der.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
    panel_der.columnconfigure(0, weight=1)
    panel_der.columnconfigure(1, weight=1)
    panel_der.rowconfigure(0, weight=1)

    # Tarjeta Izquierda: Imagen Original
    col_izq = tk.Frame(panel_der, bg="#202020", padx=10, pady=10)
    col_izq.grid(row=0, column=0, sticky="nsew", padx=6, pady=4)

    tk.Label(col_izq, text="Imagen Original f(x,y)", bg="#202020", fg="white", font=("Arial", 11, "bold")).pack()
    lbl_info_original_dim = tk.Label(col_izq, text="Esperando imagen...", bg="#202020", fg="#888888", font=("Arial", 9))
    lbl_info_original_dim.pack(pady=(0, 2))

    tk.Label(col_izq, text="Resolución nativa", bg="#202020", fg="#666666", font=("Arial", 8, "italic")).pack(pady=(0, 2))
    lbl_espacio_simetria = tk.Label(col_izq, text="Vista previa de referencia original", bg="#202020", fg="#555555", font=("Arial", 8))
    lbl_espacio_simetria.pack(pady=(0, 2))

    # Tarjeta Derecha: Imagen Reducida
    col_der = tk.Frame(panel_der, bg="#202020", padx=10, pady=10)
    col_der.grid(row=0, column=1, sticky="nsew", padx=6, pady=4)

    tk.Label(col_der, text="Imagen Reducida g(x,y)", bg="#202020", fg="#4ec9b0", font=("Arial", 11, "bold")).pack()
    lbl_info_resultado_dim = tk.Label(col_der, text="Sin procesar", bg="#202020", fg="#4ec9b0", font=("Arial", 9, "bold"))
    lbl_info_resultado_dim.pack()

    # Indicador exacto de la máscara utilizada
    lbl_info_resultado_mascara = tk.Label(col_der, text="Máscara: Ninguna procesada", bg="#202020", fg="#ffd700", font=("Arial", 9, "bold"))
    lbl_info_resultado_mascara.pack(pady=(1, 1))

    lbl_info_resultado_detalles = tk.Label(col_der, text="Ajusta los parámetros y presiona Procesar", bg="#202020", fg="#888888", font=("Arial", 8))
    lbl_info_resultado_detalles.pack(pady=(0, 2))

    # Botón interactivo para consultar la matriz aplicada
    btn_ver_mascara = tk.Button(
        col_der, text="Ver Matriz W(x,y) Aplicada", command=mostrar_matriz_aplicada,
        bg="#2d2d30", fg="#4ec9b0", font=("Arial", 8, "bold"), relief=tk.FLAT, cursor="hand2", state=tk.DISABLED
    )
    btn_ver_mascara.pack(pady=(0, 2))

    # Barra interactiva de control de Zoom (Aumentar / Disminuir / Reset)
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

    # Marco simulado en columna izquierda para alinear exactamente las dos cajas de imagen
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

    # Eventos de zoom interactivo en canvas: Rueda = Zoom, Arrastre = Mover, Doble clic = 1.0x
    canvas_resultado.bind("<MouseWheel>", on_scroll_zoom)
    canvas_resultado.bind("<Button-4>", lambda e: aplicar_delta_zoom(1.25, e.x, e.y))
    canvas_resultado.bind("<Button-5>", lambda e: aplicar_delta_zoom(1.0 / 1.25, e.x, e.y))
    canvas_resultado.bind("<ButtonPress-1>", on_drag_inicio)
    canvas_resultado.bind("<B1-Motion>", on_drag_mover)
    canvas_resultado.bind("<Double-Button-1>", resetear_zoom)

    # Alias para compatibilidad con código existente
    lbl_img_resultado = canvas_resultado

    # Por defecto cargamos la Gaussiana de clase (3x3)
    cargar_preset_gaussiana_clase()

    return ventana


def iniciar_interfaz():
    app = crear_interfaz()
    app.mainloop()


if __name__ == "__main__":
    iniciar_interfaz()
