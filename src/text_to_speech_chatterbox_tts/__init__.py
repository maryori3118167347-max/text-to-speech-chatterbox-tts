import torchaudio as ta
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

print ("cargando modelo...")

modelo = ChatterboxMultilingualTTS.from_pretrained(
    device="cpu"
)
texto ="Hola, bienvenidos a Inteligencia Artificial Generativa, crearemos audios,imagenes, textos y si el pc aguanta videos."
print("Generando audio..")
audio = modelo.generate(
    texto,
    language_id="es",
    exaggeration=0.7, #expresion o emocion
    cfg_weight=0.4,#ajusta el ritmo de la voz
    temperature=0.8,#controla la variabilidad
    repeticion_penalty=1.2,#penalizacion extrañas
    min_p=0.05,#filtra opciones durante la generacion
    top_p=1.0# controla el muestreo probabilistico
    
)
ta.save(
    "audio.wav",
    audio,
    modelo.sr
)

print("Audio guardado.")
