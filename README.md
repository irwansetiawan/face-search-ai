# Introduction

Face Search AI is a web application that analyzes and compares faces using a source image and find matching faces in the target image using machine learning, powered by AWS.

Nobody can notice Leonardo DiCaprio in a crowd, but AI can:

![preview](https://github.com/irwansetiawan/face-search-ai/assets/1826105/65bbb40d-be97-4322-a357-a2218a5d951a)

# Requirements

1. Create an AWS account
2. Create an IAM user and generate credentials, create AWS access key and AWS secret access key that you can add to `.env` file. Ensure the IAM user has full access to AWS Rekognition.

Note: Every request will incur a very small cost on your AWS account. In North Virginia region, it'll cost $0.001 for every request.

# Run in localhost

```
npm run build
npm run server
```

Then open http://localhost:3000/ in your browser.
