import json
import boto3
import pickle
import re
import os

model = None
vectorizer = None

STOP_WORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your',
    'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she',
    'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their',
    'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that',
    'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
    'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',
    'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through',
    'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
    'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then',
    'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'both',
    'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
    'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will',
    'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain',
    'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma',
    'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn'
}


def load_artifacts():
    global model, vectorizer
    s3 = boto3.client('s3')
    bucket = os.environ['MODEL_BUCKET']
    prefix = os.environ.get('MODEL_PREFIX', 'models/')

    vec_obj = s3.get_object(Bucket=bucket, Key=f'{prefix}vectorizer.pkl')
    vectorizer = pickle.loads(vec_obj['Body'].read())

    model_obj = s3.get_object(Bucket=bucket, Key=f'{prefix}model.pkl')
    model = pickle.loads(model_obj['Body'].read())
    print("Artifacts loaded from S3")


def clean_text(text):
    text = text.lower()
    text = re.sub(r'<br\s*/?>', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = [w for w in text.split() if w not in STOP_WORDS and len(w) > 2]
    return ' '.join(words)


def lambda_handler(event, context):
    global model, vectorizer

    if model is None:
        load_artifacts()

    try:
        body = event.get('body', event)
        if isinstance(body, str):
            body = json.loads(body)

        review = body.get('review', '').strip()
        if not review:
            return {
                'statusCode': 400,
                'headers': {'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*'},
                'body': json.dumps({'error': 'Missing required field: "review"'})
            }

        cleaned = clean_text(review)
        vectorized = vectorizer.transform([cleaned])
        prediction = int(model.predict(vectorized)[0])
        proba = model.predict_proba(vectorized)[0].tolist()
        confidence = float(proba[prediction])

        neutral_low = float(os.environ.get('NEUTRAL_LOW', '0.4'))
        neutral_high = float(os.environ.get('NEUTRAL_HIGH', '0.6'))

        if neutral_low <= confidence <= neutral_high:
            sentiment = 'neutral'
        else:
            sentiment = 'positive' if prediction == 1 else 'negative'

        return {
            'statusCode': 200,
            'headers': {'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*'},
            'body': json.dumps({
                'sentiment': sentiment,
                'confidence': round(confidence * 100, 2),
                'probabilities': {
                    'negative': round(proba[0] * 100, 2),
                    'positive': round(proba[1] * 100, 2)
                },
                'review_preview': review[:100] + '...' if len(review) > 100 else review
            })
        }

    except Exception as e:
        print(f"Error: {str(e)}")
        return {
            'statusCode': 500,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({'error': str(e)})
        }
