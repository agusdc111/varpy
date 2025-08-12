# Offside Line (VAR‑like) — Foto única (V3c8)

Software de escritorio (Windows) para **trazar líneas y evaluar offside desde una foto** al estilo VAR. 
Incluye **calibración por homografía**, control de **ejes X/Y** (con override) y **eje Z** (definición por punto de fuga), 
**guías** para hombro/rodilla, **zoom/pan**, **tolerancia configurable**, **detección Hough** de líneas y una 
**autodetección simple de jugadores** (HOG + k‑means). Exporta el resultado a **PNG + JSON**.

> 🛠️ Stack: Python · PySide6 · OpenCV · NumPy

---

## ✨ Características clave

- **Calibración por 4 puntos** sobre un rectángulo de referencia:
  - Área de penal *(40.32 × 16.5 m)*
  - Área chica *(18.32 × 5.5 m)*
  - Rectángulo genérico *(sin escala)*
  - Rectángulo con **escala real** (profundidad conocida, p. ej. 16.5 m)
- **Determinación del eje de offside** automática, con opción de **forzar eje X/Y**.
- **Eje Z (vertical de escena) definido por punto de fuga**:
  - Botón **Definir Z** → 4 clics: dos puntos en una vertical real + dos puntos en otra.
  - Durante la captura se ven **puntos 1..4** y **dos líneas extendidas** recortadas al frame.
  - Al finalizar, se marca **VP_Z** (cruz + etiqueta) para trazar guías verticales (p. ej., **hombro**).
- **Guías**: ⊥ Offside (auto), **Eje X**, **Eje Y**, **Eje Z** (hacia VP_Z).
- **Zoom** con rueda, **pan** con botón derecho, **grosor** de línea ajustable.
- **Slider de tolerancia** (en cm) aplicado al veredicto.
- **Detección de líneas (Hough)** para sugerir rectángulos / líneas de gol.
- **Autodetección de jugadores** (personas) con HOG; clustering simple (2 equipos) para sugerir:
  - **2.º último defensor**
  - **Atacante más adelantado** (editable manualmente)
- **Exportación** de anotaciones a **PNG** + **JSON** (puntos, parámetros y opciones usadas).

---

## 🧩 Flujo de uso en 6 pasos

1. **Abrir imagen** (menú o barra).
2. **Elegir plantilla** (Área de penal / Área chica / Genérico / Genérico con escala).
3. **Empezar calibración** → marcar 4 esquinas *(orden: lejos‑izq → lejos‑der → cerca‑der → cerca‑izq)*.
4. *(Opcional)* **Definir Z** → 4 clics en dos verticales reales para habilitar guías “hombro”.
5. **Marcar 2.º último defensor** y **Atacante** (y **Pelota** si la jugada lo requiere).
6. **Calcular offside**. Ajustar **tolerancia** y **grosor** si es necesario. **Exportar** si querés guardar.

> Tip: también podés probar **Detectar líneas (Hough)** y **Autodetectar jugadores** para acelerar la preparación.

---

## ⌨️ Controles útiles

- **Scroll**: zoom (centrado bajo el cursor).
- **Botón derecho + arrastrar**: pan.
- **Ajustar**: auto‑fit de la imagen a la vista.
- **1:1**: pixel‑perfect (100%).

---

## 🖼️ Guías y eje Z (caso “offside por hombro”)

- En el combo de guías elegí **Eje Z** y pulsá **Añadir guía**; cliqueá exactamente sobre el **hombro**.
- La guía se traza **hacia VP_Z** (o vertical de imagen si no hay VP_Z).
- También podés trazar guías en **Eje X/Y** (paralelas al eje de offside/calibración).

---

## 📦 Instalación rápida (Windows)

```powershell
# en la carpeta del proyecto
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
python offside_var_like\\main.py
```

**Requisitos** (ver `requirements.txt`):
- Python 3.10+ recomendado
- PySide6 6.7.x
- OpenCV 4.10+
- NumPy 1.26+

---

## 🧠 Cómo funciona (alto nivel)

- **Homografía**: con 4 puntos en imagen ↔ rectángulo en mundo (con o sin escala real) se estima `H` y `H⁻¹`.
- **Eje de offside (X o Y)**: se decide automáticamente comparando una recta de prueba con la **línea de gol** (arista 3→4) y, si querés, se **fuerza** desde la UI.
- **Líneas de decisión**: se proyectan en mundo (X/Y = constante) y se traen a imagen con `H⁻¹` (recortadas al frame).
- **Eje Z**: a partir de **dos verticales reales** se obtiene el **punto de fuga**; las guías Z se dibujan hacia ese VP_Z.
- **Jugadores (auto)**: detección HOG de personas + k-means (k=2) por color/posición para separar “equipos” y sugerir **2.º último defensor** y **atacante** (editable).

---

## 🧪 Casos límite y buenas prácticas

- **Paralaje / lente**: el modelo asume **plano** (campo) + **proyección pinhole**; no corrige distorsión de lente ni curvaturas.
- **VP_Z inestable**: evitá usar **verticales casi paralelas** entre sí o muy cortas para definir Z.
- **Tolerancia**: usá la tolerancia (cm) para cubrir pixelado, espesor de línea de banda y ambigüedades de marcación.
- **Escala real**: para jugadas lejos del área, “Rectángulo con escala” ayuda a estabilizar el eje y la métrica.

---

## 🗂️ Estructura del repo

```
.
├── offside_var_like/
│   ├── main.py            # UI (PySide6), overlay, eventos, guías, VP_Z, export
│   ├── offside_core.py    # homografía, ejes, proyección de líneas, veredicto
│   └── requirements.txt   # dependencias
└── README.md
```

---

## 🚧 Roadmap (ideas próximas)

- Calibración con **dos familias de paralelas** (afín) y **ajuste a escala** más robusto.
- Snapping inteligente a bandas/área mediante **detección semiautomática** de líneas.
- **Mejor detector** de jugadores (segmentación o CLIPs) y heurísticas de 2.º último defensor más fiables.
- Exportación a **proyecto reproducible** (marcas, H, opciones) para volver a abrir una jugada.
- Paquete **.exe** (PyInstaller) para Windows.

---

## 🤝 Contribuciones

¡Issues y PRs bienvenidos! Si proponés cambios en la lógica de offside, acompañá con:
- imagen de prueba,
- marcas de referencia,
- explicación del criterio y del impacto esperado.

---

## ⚖️ Licencia

Elegí la que prefieras (MIT/BSD/Apache‑2.0). Si no indicás ninguna, el repo queda **sin licencia explícita**.

---

## 📸 Créditos

Inspirado en herramientas VAR, implementado con PySide6 + OpenCV a partir de una foto única. Hecho para análisis didáctico y prototipado. No es un sistema oficial.

