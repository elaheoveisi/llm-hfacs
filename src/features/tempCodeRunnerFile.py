def get_bic_score():
    try: from pgmpy.estimators import BicScore; return BicScore
    except Exception: from pgmpy.estimators import BIC; return BIC
