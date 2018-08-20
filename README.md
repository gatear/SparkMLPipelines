## SparkMLPipelines


 A series of examples to showoff **ML pipelines**  
 ---------------------------------------------------
 
 
> Note: A **Machine Learning** pipeline is a sequence of dataset transformations and each transformation takes an input dataset and outputs the transformed dataset.  


1. **Creating an appropiate DataFrame**

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
                  
val df = movies.toDF("Name","Rating")     

```    
2. **Creating the neccesary Transformers for the ML Pipeline  

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

/ **Though we can obtain the same result with UDFs ( User Defined Functions )**/

val encoder = udf[Double,Double] { 
  case num if num > 7d => 2d
  case num if num > 5d => 1d
  case num: Double => 0d 
  }
  
```  
> With UDF you can obtain more complex transformations, while still respecting the pipeline like structure

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






