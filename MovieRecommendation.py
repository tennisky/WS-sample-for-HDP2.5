# 使用するSparkのライブラリを読み込み
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row
from pyspark.sql.functions import lit

# ローカルにある"U.ITEM"ファイルを読み込むための関数
# 各行内にある"|"ごとに区切り、文字コードを"ascii"かつ不要なUnicodeを削除して取得
def loadMovieNames():
    movieNames = {}
    with open("/tmp/U.ITEM") as f:
        for line in f:
            fields = line.split('|')
            movieNames[int(fields[0])] = fields[1].decode('ascii', 'ignore')
    return movieNames

# 各行ごとに区切ってデータとして取得するための関数
def parseInput(line):
    fields = line.value.split()
    return Row(userID = int(fields[0]), movieID = int(fields[1]), rating = float(fields[2]))

# プログラムの開始位置
if __name__ == "__main__":

    # SparkSessionの作成
    spark = SparkSession.builder.appName("MovieRecs").getOrCreate()

    # loadMovieNames関数を実行
    movieNames = loadMovieNames()

    # HDFSにある"u.data"をSparkで読み込み。RDDへ直接投入
    lines = spark.read.text("hdfs:///user/maria_dev/movie/u.data").rdd

    # 上記で読み込んだ"u.data"をparseInput関数で分割
    ratingsRDD = lines.map(parseInput)

    # データフレームへ変換し、キャッシュ
    ratings = spark.createDataFrame(ratingsRDD).cache()

    # ALS協調フィルタリングモデルの作成
    # ALSは主にレコメンデーションで使用されるモデル
    als = ALS(maxIter=5, regParam=0.01, userCol="userID", itemCol="movieID", ratingCol="rating")
    # モデルに学習を実行
    model = als.fit(ratings)

    # "userID=0"のレコードを出力
    print("\nRatings for user ID 0:")
    userRatings = ratings.filter("userID = 0")
    for rating in userRatings.collect():
        print movieNames[rating['movieID']], rating['rating']

    print("\nTop 10 recommendations:")
    
    # 100回以上評価されている"movieID"のみを抽出
    ratingCounts = ratings.groupBy("movieID").count().filter("count > 100")
    
    # 上記で作成した"userID=0"のみのレコードと100回以上評価されている"movieID"を結合
    popularMovies = ratingCounts.select("movieID").withColumn('userID', lit(0))

    # 作成したモデルおよび"userID=0"のデータを使い学習
    recommendations = model.transform(popularMovies)

    # レコメンデーションの上位10件を取得
    topRecommendations = recommendations.sort(recommendations.prediction.desc()).take(10)

    # レコメンデーションの上位10件を取得
    for recommendation in topRecommendations:
        print (movieNames[recommendation['movieID']], recommendation['prediction'])

    # Sparkを終了
    spark.stop()
