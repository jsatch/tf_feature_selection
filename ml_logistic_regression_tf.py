import tensorflow as tf
import sys
import tempfile
import random
import ag
import time

_CSV_COLUMNS = ["CODIGO", "CLUMP", "SIZE", "SHAPE","ADH", "EPIT", "BNU", "BCHR", "NNU", "MIT","CLASS"]
_CSV_COLUMN_DEFAULTS = [[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],['']]

TAM_DATASET_TRAIN = 485
TAM_DATASET_TEST = 199

def load_dataset(datafile, TAM):
  def parse_csv_line(line):
    columns = tf.decode_csv(
      line, 
      record_defaults=_CSV_COLUMN_DEFAULTS
    )
    features = dict(list(zip(_CSV_COLUMNS, columns)))
    #features.pop('CODIGO')
    #features.pop('CLUMP')
    #features.pop('ADH') 
    #features.pop('EPIT')
    #features.pop('BNU')
    #features.pop('BCHR')
    #features.pop('NNU')
    #features.pop('MIT') 
    #features.pop('SIZE')
    label = features.pop('CLASS')
    return features, label

  # Obtener el dataset en tensorflow (del archivo)
  dataset = tf.data.TextLineDataset(datafile)
  # Lo mezclamos aleatoriamente
  dataset = dataset.shuffle(buffer_size=TAM)
  
  # Transformamos el dataset a un dataset de tuplas (features, label)
  dataset = dataset.map(parse_csv_line, num_parallel_calls=5)
  dataset = dataset.repeat(100)
  dataset = dataset.batch(10)

  iterator = dataset.make_one_shot_iterator()
  features, label = iterator.get_next()

  return features, label

def build_columns(feature_selection):
  FEATURES = ["CLUMP", "SIZE", "SHAPE","ADH", "EPIT", "BNU", "BCHR", "NNU", "MIT"] 
  #FEATURES = [ "SHAPE"] 
  features_seleccionados = []
  for i in range(0, len(feature_selection)):
    if feature_selection[i] == 1:
      print(FEATURES[i])
      features_seleccionados.append(tf.feature_column.numeric_column(FEATURES[i]))
    
  return features_seleccionados

def genetico():
  def generador(max):
    especie = []
    for i in range(0, max):
      especie.append(random.randint(0,1))

    return especie
  
  # Funciones geneticas 

  def fitness(especie):
    base_columns = build_columns(especie)
    model_dir = tempfile.mkdtemp()
    model = tf.estimator.LinearClassifier(
      model_dir=model_dir, feature_columns=base_columns,
      n_classes=2,
      label_vocabulary=['2','4'],
      optimizer=tf.train.ProximalAdagradOptimizer(
              learning_rate=0.001,
              l1_regularization_strength=0.001
      )
    )

    model.train(input_fn=lambda: load_dataset('./data/wbcd_train.csv', TAM_DATASET_TRAIN))

    results = model.evaluate(input_fn=lambda: load_dataset(
      './data/wbcd_test.csv', TAM_DATASET_TEST))
    
    return results['accuracy']

  def f_reproduccion(pareja1, pareja2):
    k = random.randint(0, len(pareja1.genes))
    parte_izq = pareja1.genes[0:k]
    parte_der = pareja2.genes[k:]
    return parte_izq + parte_der

  def f_mutacion(genes):
    pos = random.randint(0, len(genes)-1)
    genes[pos] = 0 if genes[pos] == 1 else 1
    return  genes
  # Fin funciones geneticas

  POBLACION = 5
  MAX_ITERACIONES = 5
  PORCENTAJE_MUTACION = 0.1

  tiempo_inicial = time.time()
  print("Generando poblacion inicial")
  poblacion = ag.Poblacion(POBLACION, generador, fitness, f_reproduccion, f_mutacion, PORCENTAJE_MUTACION)
  for i in range(0, MAX_ITERACIONES):
    poblacion.imprimir()
    print("Tiempo: {} seg".format(time.time() - tiempo_inicial))
    print("({})=======================================".format(i))
    poblacion.seleccion()
    print("Generando poblacion (reproduccion)")
    poblacion.reproduccion()
    print("Promedio Fitness: {}".format(poblacion.promedio_fitness()))
    poblacion.mutar()

def main():
  tf.logging.set_verbosity(tf.logging.ERROR)

  base_columns = build_columns([0,0,0,1,0,0,0,0,0])
  model_dir = tempfile.mkdtemp()
  model = tf.estimator.LinearClassifier(
    model_dir=model_dir, feature_columns=base_columns,
    n_classes=2,
    label_vocabulary=['2','4'],
    optimizer=tf.train.ProximalAdagradOptimizer(
            learning_rate=0.001,
            l1_regularization_strength=0.001
    )
  )

  model.train(input_fn=lambda: load_dataset('./data/wbcd_train.csv', TAM_DATASET_TRAIN))

  results = model.evaluate(input_fn=lambda: load_dataset(
    './data/wbcd_test.csv', TAM_DATASET_TEST))
  for key in sorted(results):
      print('%s: %s' % (key, results[key]))
  
  
  #genetico()


if __name__ == '__main__':
  main()
  #with tf.Session() as sess:
    #sess.run(main=main, argv=[sys.argv[0]])
  #tf.app.run(