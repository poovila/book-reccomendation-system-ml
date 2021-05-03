from flask import Flask, render_template, flash, redirect, url_for, session, request, logging
from flask_mysqldb import MySQL
from wtforms import Form, StringField, TextAreaField, PasswordField, validators
from passlib.hash import sha256_crypt
from functools import wraps
#mod
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
import numpy as np
from sklearn.neighbors import NearestNeighbors

app = Flask(__name__)
# Config MySQL
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'admin'
app.config['MYSQL_DB'] = 'mydb'
app.config['MYSQL_CURSORCLASS'] = 'DictCursor'

# init MYSQL
mysql = MySQL(app)



@app.route('/')
def index():
        return render_template('home.html')

@app.route('/about')
def about():
        return render_template('about.html')

#def booksearch():
 #   return render_template('knn.html')
#mod
@app.route('/knn')
def knn():
        return render_template('knn.html')

@app.route('/svdindex')
def svdindex():
    return render_template('svdindex.html')
class Books:
    def __init__(self):
        self.books = pd.read_csv('./Book/Books.csv')
        self.users = pd.read_csv('./Book/Users.csv')
        self.ratings = pd.read_csv('./Book/Ratings.csv')

        # Splitting Explicit and Implicit user ratings
        # we are removing the rating set which is having the rating as 0
        self.ratings_explicit = self.ratings[self.ratings.bookRating != 0]
        self.ratings_implicit = self.ratings[self.ratings.bookRating == 0]

        # Each Books Mean ratings and Total Rating Count
        self.average_rating = pd.DataFrame(
            self.ratings_explicit.groupby('ISBN')['bookRating'].mean())
        self.average_rating['ratingCount'] = pd.DataFrame(
            self.ratings_explicit.groupby('ISBN')['bookRating'].count())
        self.average_rating = self.average_rating.rename(
            columns={'bookRating': 'MeanRating'})

        # To get a stronger similarities
        counts1 = self.ratings_explicit['userID'].value_counts()
        self.ratings_explicit = self.ratings_explicit[
            self.ratings_explicit['userID'].isin(counts1[counts1 >= 50].index)]

        # Explicit Books and ISBN
        self.explicit_ISBN = self.ratings_explicit.ISBN.unique()
        self.explicit_books = self.books.loc[self.books['ISBN'].isin(
            self.explicit_ISBN)]

        # Look up dict for Book and BookID
        self.Book_lookup = dict(
            zip(self.explicit_books["ISBN"], self.explicit_books["bookTitle"]))
        self.ID_lookup = dict(
            zip(self.explicit_books["bookTitle"], self.explicit_books["ISBN"]))

    def Top_Books(self, n=10, RatingCount=100, MeanRating=3):
        # here we are specifying the latency of meanRating with value of 3
        # and latency of RatingCount with value of 100
        # this makes a threshold value for predicting the best possible book sets for the user
        # books with the highest rating
        # this function will not recommend any books just shows the highest rated books rated by every user
        BOOKS = self.books.merge(self.average_rating, how='right', on='ISBN')
        # print(Books)
        M_Rating = BOOKS.loc[BOOKS.ratingCount >= RatingCount].sort_values(
            'MeanRating', ascending=False).head(n)

        H_Rating = BOOKS.loc[BOOKS.MeanRating >= MeanRating].sort_values(
            'ratingCount', ascending=False).head(n)

        # print(M_Rating)
        # print(H_Rating)

        return M_Rating, H_Rating


class KNN(Books):

    def __init__(self, n_neighbors=5):
        # calling super class __init__ method
        super().__init__()
        # assigning k  value = 5
        self.n_neighbors = n_neighbors
        # removing nan value
        self.ratings_mat = self.ratings_explicit.pivot(
            index="ISBN", columns="userID", values="bookRating").fillna(0)
        '''
        Implementing kNN
        In numerical analysis and scientific computing, a sparse matrix or sparse array is a matrix in which
        most of the elements are zero.
        We convert our table to a 2D matrix, and fill the missing values with zeros 
        (since we will calculate distances between rating vectors). We then transform the values(ratings)
        of the matrix dataframe into a scipy sparse matrix for more efficient calculations.
        Finding the Nearest Neighbors We use unsupervised algorithms with sklearn.neighbors.
        The algorithm we use to compute the nearest neighbors is “brute”, and we specify “metric=cosine” 
        algorithm will calculate the cosine similarity between rating vectors. Finally, we fit the model.
        '''
        self.uti_mat = csr_matrix(self.ratings_mat.values)
        # KNN Model Fitting
        # KNN Model Fitting
        # using cosine similarity
        '''Mathematically, it measures 
        the cosine of the angle between two vectors projected in a multi-dimensional space
        Cosine similarity is a metric used to determine how
         similar the documents are irrespective of their size.'''
        self.model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
        self.model_knn.fit(self.uti_mat)

    def Recommend_Books(self, book, n_neighbors=5):
        # Book Title  to BookID
        # bID = list(self.Book_lookup.keys())[list(self.Book_lookup.values()).index(book)]
        bID = self.ID_lookup[book]

        query_index = self.ratings_mat.index.get_loc(bID)

        KN = self.ratings_mat.iloc[query_index, :].values.reshape(1, -1)

        distances, indices = self.model_knn.kneighbors(
            KN, n_neighbors=n_neighbors + 1)

        Rec_books = list()
        Book_dis = list()

        for i in range(1, len(distances.flatten())):
            Rec_books.append(self.ratings_mat.index[indices.flatten()[i]])
            Book_dis.append(distances.flatten()[i])

        Book = self.Book_lookup[bID]

        Recommmended_Books = self.books[self.books['ISBN'].isin(Rec_books)]

        return Book, Recommmended_Books, Book_dis


@app.route('/predict', methods=['POST'])
def predict():
    global KNN_Recommended_Books
    if request.method == 'POST':
        ICF = KNN()
        book = request.form['book']
        data = book

        _, KNN_Recommended_Books, _ = ICF.Recommend_Books(data)

        KNN_Recommended_Books = KNN_Recommended_Books.merge(
            ICF.average_rating, how='left', on='ISBN')
        KNN_Recommended_Books = KNN_Recommended_Books.rename(
            columns={'bookRating': 'MeanRating'})

        df = pd.DataFrame(KNN_Recommended_Books, columns=['bookTitle', 'bookAuthor', 'MeanRating'])

    return render_template('result.html', predictionB=KNN_Recommended_Books[['bookTitle']],
                           predictionA=KNN_Recommended_Books[['bookAuthor']],
                           predictionR=KNN_Recommended_Books[['MeanRating']],
                           prediction=df)

#mod-complete



#Register form class
class RegisterForm(Form):
        name = StringField('Name', [validators.Length(min=1, max=50)])

        username = StringField('Username', [validators.Length(min=4, max=25)])

        email = StringField('Email', [validators.Length(min=6, max=50)])

        password = PasswordField('Password', [

                validators.DataRequired(),

                validators.EqualTo('confirm', message='Passwords do not match')
    ])

        confirm = PasswordField('Confirm Password')


# User Register
@app.route('/register', methods = ['GET', 'POST'])
def register():
        form = RegisterForm(request.form)
        if request.method == 'POST' and form.validate():
                name = form.name.data
                email = form.email.data
                username = form.username.data
                password = sha256_crypt.encrypt(str(form.password.data))

                # create cursor
                cur = mysql.connection.cursor()

                # Execute query
                cur.execute("INSERT INTO register(name, email, username, password) VALUES(%s, %s, %s, %s)", (name, email, username, password))

                # Commit to DB
                mysql.connection.commit()

                # Close connection
                cur.close()

                flash('You are now registered and can log in', 'success')

                return redirect(url_for('index'))
        return render_template('register.html', form=form)

@app.route('/login', methods=['GET', 'POST'])

def login():
        if request.method == 'POST':
                # Get Form Fields

                username = request.form['username']

                password_candidate = request.form['password']

                # Create cursor

                cur = mysql.connection.cursor()
                # Get user by username

                result = cur.execute("SELECT * FROM register WHERE username = %s", [username])


                if result > 0:
                        # Get stored hash

                        data = cur.fetchone()
                        password = data['password']


                        # Compare Passwords

                        if sha256_crypt.verify(password_candidate, password):
                                session['logged_in'] = True

                                session['username'] = username


                                flash('You are now logged in', 'success')

                                return redirect(url_for('about'))
                        else:
                                error = 'Invalid login'
                                return render_template('login.html', error=error)
                        cur.close()
                else:
                        error="username not found"
                        return render_template('search.html',error=error)
        return render_template('login.html')

# check if user logged in:
def is_logged_in(f):
     @wraps(f)
     def wrap(*args, **kwargs):
          if 'logged_in' in session:
               return f(*args,**kwargs)
          else:
               flash('Unauthorized, Please Login','danger')
               return redirect(url_for('login'))
     return wrap 

@app.route('/logout')
@is_logged_in
def logout():
     session.clear()
     flash('You are logged out','success')
     return redirect(url_for('login'))

@app.route('/dashboard')
@is_logged_in
def dashboard():
         return render_template('sample.html')  


     
if __name__ == "__main__":
     app.secret_key = 'secret123'
     app.run(debug = True)

