## Fast API Reference

To start server run following command

```
uvicorn routes.api:app
```

There are two API calls:

- detect_emotion_full (Returns from a list of emotions)
- detect_emotion_binary (Return 'positive' or 'negative' as class)

### Sample API calls

```
http POST http://localhost:8000/detect_emotion_binary text="Weird things are happening"
http POST http://localhost:8000/detect_emotion_full text="This looks nice"
```
