from keras.models import load_model

model = load_model("../kickstarter3/data/my_model.h5")

model.compile(model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['mse']))

# given imputs take test df
def user_input(user_campaign):
    X_features = user_campaign[['launch_to_deadline_days', 'staff_pick', 'pledged', 'backers_count', 'spotlight', 'goal']]
    return model.predict(X_features)

user_input
