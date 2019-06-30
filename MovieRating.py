# 使用するSparkのライブラリを読み込み
from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql import functions

# ローカルにある"U.ITEM"ファイルを読み込むための関数
# 各行内にある"|"ごとに区切って取得
def loadMovieNames():
    movieNames = {}
    with open("/tmp/U.ITEM") as f:
        for line in f:
            fields = line.split('|')
            movieNames[int(fields[0])] = fields[1]
    return movieNames

# 各行ごとに区切ってデータとして取得するための関数
def parseInput(line):
    fields = line.split()
    return Row(movieID = int(fields[1]), rating = float(fields[2]))

# プログラムの開始位置
if __name__ == "__main__":

    # SparkSessionの作成
    spark = SparkSession.builder.appName("PopularMovies").getOrCreate()

    # loadMovieNames関数を実行
    movieNames = loadMovieNames()

    # HDFSにある"u.data"をSparkで読み込み
    lines = spark.sparkContext.textFile("hdfs:///user/maria_dev/movie/u.data")
    
    # 上記で読み込んだ"u.data"をparseInput関数で分割してRDDへデータを投入
    movies = lines.map(parseInput)
    
    # データフレームへ変換
    movieDataset = spark.createDataFrame(movies)

    # "movieID"ごとに平均の"rating"を取得
    averageRatings = movieDataset.groupBy("movieID").avg("rating")

    # "movieID"ごとにデータの個数をカウント
    counts = movieDataset.groupBy("movieID").count()

    # 取得したデータを１つのデータに統合
    averagesAndCounts = counts.join(averageRatings, "movieID")

    # 評価が下位10件を取得
    bottomTen = averagesAndCounts.orderBy("avg(rating)").take(10)

    # 取得した10件を表示
    for movie in bottomTen:
        print (movieNames[movie[0]], movie[1], movie[2])

    # Sparkを終了
    spark.stop()
