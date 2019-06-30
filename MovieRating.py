# �g�p����Spark�̃��C�u������ǂݍ���
from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql import functions

# ���[�J���ɂ���"U.ITEM"�t�@�C����ǂݍ��ނ��߂̊֐�
# �e�s���ɂ���"|"���Ƃɋ�؂��Ď擾
def loadMovieNames():
    movieNames = {}
    with open("/tmp/U.ITEM") as f:
        for line in f:
            fields = line.split('|')
            movieNames[int(fields[0])] = fields[1]
    return movieNames

# �e�s���Ƃɋ�؂��ăf�[�^�Ƃ��Ď擾���邽�߂̊֐�
def parseInput(line):
    fields = line.split()
    return Row(movieID = int(fields[1]), rating = float(fields[2]))

# �v���O�����̊J�n�ʒu
if __name__ == "__main__":

    # SparkSession�̍쐬
    spark = SparkSession.builder.appName("PopularMovies").getOrCreate()

    # loadMovieNames�֐������s
    movieNames = loadMovieNames()

    # HDFS�ɂ���"u.data"��Spark�œǂݍ���
    lines = spark.sparkContext.textFile("hdfs:///user/maria_dev/movie/u.data")
    
    # ��L�œǂݍ���"u.data"��parseInput�֐��ŕ�������RDD�փf�[�^�𓊓�
    movies = lines.map(parseInput)
    
    # �f�[�^�t���[���֕ϊ�
    movieDataset = spark.createDataFrame(movies)

    # "movieID"���Ƃɕ��ς�"rating"���擾
    averageRatings = movieDataset.groupBy("movieID").avg("rating")

    # "movieID"���ƂɃf�[�^�̌����J�E���g
    counts = movieDataset.groupBy("movieID").count()

    # �擾�����f�[�^���P�̃f�[�^�ɓ���
    averagesAndCounts = counts.join(averageRatings, "movieID")

    # �]��������10�����擾
    bottomTen = averagesAndCounts.orderBy("avg(rating)").take(10)

    # �擾����10����\��
    for movie in bottomTen:
        print (movieNames[movie[0]], movie[1], movie[2])

    # Spark���I��
    spark.stop()
