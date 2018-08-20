## SparkMLPipelines


 A series of examples to showoff **ML pipelines**  
 ---------------------------------------------------
 
 
> Note: A **Machine Learning** pipeline is a sequence of dataset transformations and each transformation takes an input dataset and outputs the transformed dataset.  


1. **Creating an appropiate DataFrame :**

```Scala

val movies = Seq( ("The Godfather", 9.2d),
                  ("The Shawshank Redemption", 9.3d),
                  ("The Godfather: Part 2", 9d),
                  ("Inception", 8.8d),
                  ("The Dark Knight", 9d),
                  ("Monster a-Go-Go", 2.3d),
                  ("Birdemic: Shock and Terror", 1.8d),
                  ("Manos: The Hands of Fate", 1.9d),
                  ("Batman & Robin", 3.7d),
                  ("Troll 2", 2.8d),
                  ("The Back-up Plan", 5.3d),
                  ("I Love You, Man", 7d),
                  ("Wedding Crashers", 7d) )
                  
val df = movies.toDF("Name","Rating")      :

```    
2. **A simple Transformer :** 

```Scala
                  
import org.apache.spark.ml.feature.Binarizer

val bin = new Binarizer()
  .setInputCol("Rating")
  .setOutputCol("Label")
  .setThreshold(5d)
  
 /** The transformer will output another DataFrame with an appended column 'Label' **/
 
scala> bin.transform(df).show()

+--------------------+------+-----+
|                Name|Rating|Label|
+--------------------+------+-----+
|       The Godfather|   9.2|  1.0|
|The Shawshank Red...|   9.3|  1.0|
|The Godfather: Pa...|   9.0|  1.0|
|           Inception|   8.8|  1.0|
|     The Dark Knight|   9.0|  1.0|
|     Monster a-Go-Go|   2.3|  0.0|
|Birdemic: Shock a...|   1.8|  0.0|
|Manos: The Hands ...|   1.9|  0.0|
|      Batman & Robin|   3.7|  0.0|
|             Troll 2|   2.8|  0.0|
|    The Back-up Plan|   5.3|  1.0|
|     I Love You, Man|   7.0|  1.0|
|    Wedding Crashers|   7.0|  1.0|
+--------------------+------+-----+

/**Though we can obtain the same result with UDFs ( User Defined Functions )**/

val encoder = udf[Double,Double] { 
  case num if num > 7d => 2d
  case num if num > 5d => 1d
  case num: Double => 0d 
  }
  
```
3. **Chaining Transformers into a ML Pipeline**

```Scala

/** Configure pipeline stages  **/
val binTF = Binarizer(inputCol="Rating", outputCol="Group", threshold= 5d) 
val hasTF = HashingTF(inputCol="Group", outputCol="Count", numFeatures=200) 
val vasTF = VectorAssembler( inputCols=Array("Rating","Group"), outputCol="Features")

val lr = LogisticRegression(maxIter=10, regParam=0.01) 

val pipeline = Pipeline( binTF::hasTF::vasTF::lr::Nil ) 

/**Now we can obtain an Estimator**/

```
  
> With UDFs and 'withColumn(...)' you can obtain the same result. 
> That's to be expected because Transformers use the same technique internally.

```Scala
val labeledDF = df.withColumn("Label", encoder( $"Rating" ))

scala> labeledDF.show
+--------------------+------+-----+
|                Name|Rating|Label|
+--------------------+------+-----+
|       The Godfather|   9.2|  2.0|
|The Shawshank Red...|   9.3|  2.0|
|The Godfather: Pa...|   9.0|  2.0|
|           Inception|   8.8|  2.0|
|     The Dark Knight|   9.0|  2.0|
|     Monster a-Go-Go|   2.3|  0.0|
|Birdemic: Shock a...|   1.8|  0.0|
|Manos: The Hands ...|   1.9|  0.0|
|      Batman & Robin|   3.7|  0.0|
|             Troll 2|   2.8|  0.0|
|    The Back-up Plan|   5.3|  1.0|
|     I Love You, Man|   7.0|  1.0|
|    Wedding Crashers|   7.0|  1.0|
+--------------------+------+-----+

```






