from sklearn.metrics.pairwise import cosine_similarity

def compare(face_embedding, db_embedding):

    sim = cosine_similarity(face_embedding, db_embedding)

    return sim[0][0]