# �g�p����Spark�̃��C�u������ǂݍ���
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row
from pyspark.sql.functions import lit

# ���[�J���ɂ���"U.ITEM"�t�@�C����ǂݍ��ނ��߂̊֐�
# �e�s���ɂ���"|"���Ƃɋ�؂�A�����R�[�h��"ascii"���s�v��Unicode���폜���Ď擾
def loadMovieNames():
    movieNames = {}
    with open("/tmp/U.ITEM") as f:
        for line in f:
            fields = line.split('|')
            movieNames[int(fields[0])] = fields[1].decode('ascii', 'ignore')
    return movieNames

# �e�s���Ƃɋ�؂��ăf�[�^�Ƃ��Ď擾���邽�߂̊֐�
def parseInput(line):
    fields = line.value.split()
    return Row(userID = int(fields[0]), movieID = int(fields[1]), rating = float(fields[2]))

# �v���O�����̊J�n�ʒu
if __name__ == "__main__":

    # SparkSession�̍쐬
    spark = SparkSession.builder.appName("MovieRecs").getOrCreate()

    # loadMovieNames�֐������s
    movieNames = loadMovieNames()

    # HDFS�ɂ���"u.data"��Spark�œǂݍ��݁BRDD�֒��ړ���
    lines = spark.read.text("hdfs:///user/maria_dev/movie/u.data").rdd

    # ��L�œǂݍ���"u.data"��parseInput�֐��ŕ���
    ratingsRDD = lines.map(parseInput)

    # �f�[�^�t���[���֕ϊ����A�L���b�V��
    ratings = spark.createDataFrame(ratingsRDD).cache()

    # ALS�����t�B���^�����O���f���̍쐬
    # ALS�͎�Ƀ��R�����f�[�V�����Ŏg�p����郂�f��
    als = ALS(maxIter=5, regParam=0.01, userCol="userID", itemCol="movieID", ratingCol="rating")
    # ���f���Ɋw�K�����s
    model = als.fit(ratings)

    # "userID=0"�̃��R�[�h���o��
    print("\nRatings for user ID 0:")
    userRatings = ratings.filter("userID = 0")
    for rating in userRatings.collect():
        print movieNames[rating['movieID']], rating['rating']

    print("\nTop 10 recommendations:")
    
    # 100��ȏ�]������Ă���"movieID"�݂̂𒊏o
    ratingCounts = ratings.groupBy("movieID").count().filter("count > 100")
    
    # ��L�ō쐬����"userID=0"�݂̂̃��R�[�h��100��ȏ�]������Ă���"movieID"������
    popularMovies = ratingCounts.select("movieID").withColumn('userID', lit(0))

    # �쐬�������f�������"userID=0"�̃f�[�^���g���w�K
    recommendations = model.transform(popularMovies)

    # ���R�����f�[�V�����̏��10�����擾
    topRecommendations = recommendations.sort(recommendations.prediction.desc()).take(10)

    # ���R�����f�[�V�����̏��10�����擾
    for recommendation in topRecommendations:
        print (movieNames[recommendation['movieID']], recommendation['prediction'])

    # Spark���I��
    spark.stop()
