
#%%
from tkinter import *
from tkinter.filedialog import askopenfile, asksaveasfile

from minisom import MiniSom
import numpy as np
from distinctipy import distinctipy

from PIL import Image, ImageTk

def get_image_features(image):
  """
    Devuelve un vector de dimensión 1 por cada banda (cuya longitud = alto x ancho)
    Cada posición tiene la intensidad de un píxel en una banda determinada, normalizada.
  """
  features = image.reshape((-1, image.shape[2]))
  return (features - np.mean(features, axis = 0)) / np.std(features, axis = 0)

def segment_image(image, labels, bit_depth = 8):
  """
    Función para segmentar una imágen dado una lista con labels por píxel.
    image: Imágen a segmentar.
    labels: Lista de labels por cada píxel.
    bit_depth: Profundidad de la imágen en bits.
  """
  n_clusters = len(np.unique(labels)) # El número de clusters es la cantidad de labels distintos.
  max_value = 2 ** bit_depth - 1 # Valor de intensidad máximo de la imágen según profundidad de bits
  
  # Obtengo una lista de colores al azar únicos y diferenciados rgb utilizando distinctipy, uno por cada cluster
  colors = (np.array(distinctipy.get_colors(n_clusters)) * max_value).astype(f"uint{bit_depth}")

  # Creo una matriz de resultados
  segmented = np.zeros((image.shape[0], image.shape[1], colors.shape[1]))
  # Reshape para que los labels tengan la forma de la imágen original
  labels = labels.reshape((image.shape[0], image.shape[1]))
  # Por cada píxel reemplazo el label por el color correspondiente para la clase obtenido de la lista colors.
  for i in range(labels.shape[0]):
    for j in range(labels.shape[1]):
      segmented[i, j] = colors[labels[i,j]]

  # Casteo imágen segmentada a uint segun profundidad de bits
  return segmented.astype(f"uint{bit_depth}")

def som_predict(som, features):
  """ 
    Devuelve los labels para cada feature realizando una predicción con una som entrenada, es decir, el número de cluster por cada fila.
    som: La red de Kohonen entrenada a usar.
    features: El vector de features a etiquetar.
  """
  winners = np.array([som.winner(f) for f in features]) # Obtengo coordenadas de la neurona ganadora por fila
  # Obtengo un diccionario de coordenada de neurona - clase, utilizando enumerate sobre las coordenadas posibles.
  winners_values = dict((tuple(k),v) for v,k in enumerate(np.unique(winners, axis=0)))
  # Por cada coordenada ganadora devuelvo la clase (label)
  return np.array([winners_values[tuple(w)] for w in winners])

def kohonen(image, n, m, sigma=1, learning_rate=.5, num_iterations=1000):
  """
    Entrena una red de Kohonen de n x m sobre una imágen y la segmenta segun los clusters encontrados.
    Muestra y grafica la matríz U de la red resultante.

    image: Imágen a segmentar mediante Kohonen.
    n: Cantidad de filas de la red de Kohonen.
    m: Cantidad de columnas de la red de Kohonen.
    sigma: Radio de vecindad inicial.
    learning_rate: Ratio inicial de aprendizaje. Determina cuan rápido se actualizan los pesos. A menor learning_rate, la magnitud de actualización de pesos en cada iteración es menor.
    num_iterations: Número máximo de iteraciones en caso de no llegar a una convergencia.
  """

  # Obtengo los features
  features = get_image_features(image)

  # Creo la red SOM y la entreno con los features
  som = MiniSom(n, m, features.shape[1], sigma=sigma, learning_rate=learning_rate, activation_distance='euclidean',
              topology='rectangular', neighborhood_function='gaussian', random_seed=8)
  som.train(features, num_iterations, verbose=True)

  # Obtengo los labels
  labels = som_predict(som, features)

  # Devuelvo la imágen segmentada
  return segment_image(image, labels)

class Interface():
    """
        Ésta clase se encarga de construir la interfaz de la aplicación y
        manejar los eventos producidos al interactuar con ella.
    """

    def __init__(self):
        """
            Crea e inicializa la ventana y los elementos de la aplicación.
        """
        self.bg = "#99CCFF" # Color de fondo de toda la app

        self.interface = Tk()
        self.interface.title("Trabajo Práctico N1: Rodrigo Fondato")
        self.interface.configure(bg=self.bg)
        self.image = None
        self.image_path = None
        self.original_image = None
        self._create_elements()

    def _create_elements(self):
        """
            Crea los elementos visuales de la aplicación
        """
        # Título principal
        self.title_lbl = Label(self.interface, bg=self.bg, font=("Arial", 25), text="Segmentación por Kohonen")
        self.title_lbl.pack(pady=10)

        # Etiqueta de estado: Inicialmente vacía
        self.status_lbl = Label(self.interface, bg=self.bg, fg="red", font=("Arial", 14))
        self.status_lbl.pack(pady=5)

        self._create_img_container()
        self._create_parameters()
        self._create_buttons()

    def _create_img_container(self):
        """
            Crea el marco para contener imágenes a segmentar
        """
        # Marco con bordes para contener la imágen
        self.img_container_frame = Frame(self.interface, bg=self.bg, relief="groove", borderwidth=2)
        # El contenedor de la imágen es un label que inicialmente tiene el texto "NO HAY IMAGEN"
        self.img_container = Label(self.img_container_frame,
                                   bg=self.bg, 
                                   text="NO HAY IMAGEN",
                                   padx=50,
                                   pady=50)
        self.img_container.pack()
        self.img_container_frame.pack(pady=10)

    def _create_parameters(self):
        """
            Crea la sección de parámetros para configurar la red de Kohonen.
        """
        # Marco para contener a los parámetros
        self.parameters_frame = Frame(self.interface, bg=self.bg)

        # Los parámetros se configurarán en una grilla horizontal (1 row, múltiples columnas)
        
        # Label para el parámetro Ancho SOM
        self.grid_x_lbl = Label(self.parameters_frame, bg=self.bg, text="Ancho SOM: ")
        self.grid_x_lbl.grid(row=0, column=0, padx=10)

        # Slider para Ancho SOM, inicializado en 1. Admite valores de 1 a 5.
        self.grid_x = Scale(self.parameters_frame, from_=1, to=5, orient=HORIZONTAL, bg=self.bg)
        self.grid_x.set(1)
        self.grid_x.grid(row=0, column=1, padx=10)

        # Label para el parámetro Alto SOM
        self.grid_y_lbl = Label(self.parameters_frame, bg=self.bg, text="Alto SOM: ")
        self.grid_y_lbl.grid(row=0, column=2, padx=10)

        # Slider para Alto SOM, inicializado en 1. Admite valores de 1 a 5.
        self.grid_y = Scale(self.parameters_frame, from_=1, to=5, orient=HORIZONTAL, bg=self.bg)
        self.grid_y.set(1)
        self.grid_y.grid(row=0, column=3, padx=10)

        self.parameters_frame.pack(pady=10)

    def _create_buttons(self):
        """
            Crea la botonera con todas las acciones posibles: Abrir, Segmentar, Restaurar y Grabar.
        """

        # Marco para la botonera
        self.buttons_frame = Frame(self.interface, bg=self.bg)

        # Los botones se configurarán en una grilla horizontal (1 row, múltiples columnas)

        # Botón de abrir archivo
        self.open_btn = Button(self.buttons_frame, text="Abrir", command=self._open_file)
        self.open_btn.grid(row=0, column=0, padx=10)

        # Botón de segmentar imágen
        self.segment_btn = Button(self.buttons_frame, text="Segmentar", command=self._segment)
        self.segment_btn.grid(row=0, column=1, padx=10)

        # Botón de recuperar imágen
        self.restore_btn = Button(self.buttons_frame, text="Restaurar", command=self._restore)
        self.restore_btn.grid(row=0, column=2, padx=10)

        # Botón de grabar imágen
        self.save_btn = Button(self.buttons_frame, text="Grabar", command=self._save)
        self.save_btn.grid(row=0, column=3, padx=10)

        self.buttons_frame.pack(pady=10)

    def _open_file(self):
        """
            Ejecuta el comando de abrir imágen. Muestra un diálogo donde el usuario debe elegir un archivo de imágen
            a abrir. Si no elige ninguno muestra un mensaje de estado de "Ninguna imágen fue seleccionada"
            Si elige una imágen válida, la misma es cargada en el contenedor de imágenes en pantalla.
        """
        try:
            # Abre diálogo y obtiene un archivo a cargar
            file = askopenfile(mode ='r', filetypes =[('Image Files', ['*.png', '*.jpg', '*.jpeg', '*.tiff', '*.bmp'])])
            if file is not None:
                # Si el usuario eligió un archivo válido carga la imágen y la dibuja
                self.image_path = file.name
                self.image = Image.open(file.name)
                self.original_image = self.image.copy()
                self._draw_image()
                self.status_lbl.config(text="") # Vacía el mensaje de estado
                file.close()
            else: # Sino muestra un mensaje que informa que nada fue seleccionado
                self.status_lbl.config(text="Ninguna imágen fue seleccionada")
        except Exception as e:
            # Si hay un error imprime el mensaje por consola y actualiza el estado
            print(e)
            self.status_lbl.config(text="Hubo un error al abrir la imágen")          

    def _segment(self):
        """
            Ejecuta el comando de segmentar la imágen elegida.
            Valida que haya una imágen o arroja un error en la barra de estado.
            Segmenta la imágen utiizando la red de Kohonen con los parámetros elegidos.
            Dibuja la imágen segmentada.
        """
        # Si la validacion falla, no hace nada
        if not self._validate():
           return
        
        # Actualiza el estado con el mensaje que indica que está procesando
        self.status_lbl.config(text=f"Segmentando utilizando Kohonen ({self.grid_x.get()}x{self.grid_y.get()}), por favor aguardar...")
        self.interface.update() # Para forzar un refresh de la pantalla
        # Realiza la segmentación con la red de Kohonen
        segmented = kohonen(image=np.array(self.original_image), n=self.grid_x.get(), m=self.grid_y.get())
        # Convierte el array segmentado en una imágen y la dibuja en la app
        self.image = Image.fromarray(segmented)
        self._draw_image()
        # Informa que la imágen ya fue segmentada
        self.status_lbl.config(text="Imágen Segmentada")

    def _validate(self):
        """
            Si no hay ninguna imágen seleccionada pide al usuario que lo haga,
            utilizando la barra de estado.
        """
        if self.image is None:
           self.status_lbl.config(text="Por favor elija una imágen antes de segmentar")
           return False
        
        return True
        
    def _draw_image(self):
        """
            Dibuja una imágen en el contenedor de imágenes (UI)
            Si la imágen tiene un alto mayor a 600 píxeles, la reduce para que tenga como máximo ese alto,
            modificando en la misma proporción el ancho.
        """
        im = self.image
        # Como máximo admitir 600px de alto y sino hacer un resize manteniendo proporciones.
        if im.size[1] > 600:
           proportion = im.size[0] / im.size[1]
           im = self.image.resize((int(proportion * 600), 600))

        # Convertir a una imágen de TK, y actualizar el label de img_container poniendo la imágen.
        ph = ImageTk.PhotoImage(im)
        self.img_container.config(image=ph, text=None, padx=0, pady=0)
        self.img_container.image = ph # Para mantener la referencia a ph

    def _restore(self):
       """
            Recupera la imágen original, descartando la segmentada.
            Si ninguna imágen fue seleccionada, no hace nada.
       """
       if self.original_image is None:
          return
       
       # Utiliza la referencia a la imágen original para recuperarla.
       self.image = self.original_image.copy()
       self._draw_image()
       self.status_lbl.config(text="Imágen Restaurada") # Actualiza la barra de estado

    def _save(self):
       """
            Ejecuta el comando de grabar una imágen. Muestra un cuadro de diálogo donde el usuario puede elegir
            el archivo destino y luego graba la imágen actualmente mostrada en la UI en disco.
            Si ninguna imágen fue seleccionada, no hace nada.
       """
       if self.image is None:
          return
       
       try:
          # Abre diálogo y obtiene un archivo en modo escritura para grabar la imágen
          file = asksaveasfile(mode='w', filetypes =[('Image Files', ['*.png', '*.jpg', '*.jpeg', '*.tiff', '*.bmp'])], defaultextension='*.png')

          if file is not None:
            # Si el usuario eligió un path válido, escribe el archivo con la imágen actual
            self.image.save(file.name)
            file.close() # Cierra el archivo
            self.status_lbl.config(text="Imágen grabada") # Actualiza el estado
       except Exception as e: # Si hay un error imprimo el mismo por consola y actualizo el estado
          print(e) 
          self.status_lbl.config(text="Hubo un error al grabar la imágen")
       
    def start(self):
        """
            Inicia la aplicación. Muestra la interfaz.
        """
        self.interface.mainloop()

#  Creo la interfaz gráfica y la inicio.
interface = Interface()
interface.start()

# %%
