# Penman
Howdy! This is a web application I created on my own time to solve simple handwritten equations, e.g. $2+2=$. The goal of this application was to learn technologies I wasn't very familiar with, and get comfortable knowing the ins and outs of model development into deployment.

## Tech stack
For the Convolutional Neural Network (CNN), I initially created it from scratch with just NumPy to familiarize myself with the ins and outs of a CNN, then developed an ONNX template file to export the model.  Eventually, I moved to **PyTorch** for faster fine-tuning and understanding of PyTorch's abstraction for creating models (it makes everything so much easier).

Once the CNN was fully developed, I created my own validation dataset (drawing math tokens over and over) on TLDraw (library mentioned later) and tested their accuracies pre and post fine-tuning, since the model's training dataset was slightly different in terms of stroke length and width. I then logged their accuracies with **MLFlow** to obtain the best model.

For the full-stack framework, I opted to use **Next.js and Flask**.
1. The Next.js framework allowed me to utilize TLDraw, a canvas library that makes implementing features onto one very easy.
2. Flask helped with setting up a quick and easy backend for my main inference endpoint.

Regarding the database, I opted to use **Amazon's S3 buckets**, just because I haven't used them before.

Now for moving the code to production, I initially opted to just use an Amazon EC2 instance, but upgraded to **using ECS to deploy my containerized images stored on ECR**. Utilizing these technologies made it relatively simple to flesh out a **CI/CD pipeline** (if you ignore the one million policies I had to add to my AWS IAM role), since I just configured the pipeline details with a `.yml` file with **GitHub Actions**.
## Running the project
Unfortunately since the project is integrated with my AWS credentials and bucket names, it's not useable unless you have the same set up configured.

Once you get the project running by:
1. Setting up the `.venv` in `~/backend`
2. Installing all required dependencies with `npm i ` in `~/frontend`
3. Creating an S3 bucket and changing the `penman-lln` in the cloned code to your bucket name

You should be able to run both the frontend and backend with the following commands:
1. `python app.py` for the backend
2. `npm run dev` for the backend
