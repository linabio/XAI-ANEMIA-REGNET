# Inicio - Organización del proyecto

Bienvenido. Esta guía rápida explica dónde empezar en el repositorio reorganizado.

Estructura principal creada:

- `1-datos/` — crudos, procesados, externos, validación
- `2-experimentos/` — notebooks y análisis
- `3-entrenamiento/` — configuraciones, scripts y modelos guardados
- `4-evaluacion/` — resultados, métricas, gráficos
- `5-despliegue/` — API, interfaz web y monitoreo
- `6-utilidades/` — procesamiento reutilizable, arquitecturas, herramientas, configuración
- `7-documentacion/` — guías, referencia y ejemplos
- `8-pruebas/` — pruebas rápidas, completas y datos de prueba

Siguientes pasos recomendados:
1. Revisar `3-entrenamiento/configuracion` y ajustar los YAML de entrenamiento si es necesario.
2. Ejecutar los scripts de entrenamiento en `3-entrenamiento/scripts` para verificar rutas de modelos.
3. Probar la interfaz en `5-despliegue/interfaz-web/app.py`.
4. Añadir tests en `8-pruebas/` y ejecutar con `pytest`.

Si necesitas que mueva archivos adicionales o adapte scripts a las nuevas rutas, dime qué quieres que haga a continuación.