from flask import Flask, render_template, request
import webscrape, sentanalysis
import pandas as pd


app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        input_value = request.form['inputField']
        s = webscrape.scrapfyt(input_value)

        sentiment = sentanalysis.sentiment_analysis("Full Comments.csv");
        

        list_file_and_detail = list(s)
        list_sentiment = list(sentiment)
        print(list_file_and_detail)
        video_title, video_owner, video_comment_with_replies, video_comment_without_replies = list_file_and_detail[1:]
        pos_comments_csv, neg_comments_csv, video_positive_comments, video_negative_comments = list_sentiment
        pos_comments_csv = pd.read_csv('Positive Comments.csv')
        neg_comments_csv = pd.read_csv('Negative Comments.csv')
        
        
        return render_template('result.html', input_value=input_value, op=s, sent=sentiment, title = video_title,
                           owner = video_owner, comment_w_replies = video_comment_with_replies,
                           comment_wo_replies = video_comment_without_replies,
                           positive_comment = video_positive_comments, negative_comment = video_negative_comments,
                           pos_comments_csv = [pos_comments_csv.to_html()], neg_comments_csv = [neg_comments_csv.to_html()])

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
