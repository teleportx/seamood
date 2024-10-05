"""Module made for working with recommender systems made via AI."""

from collections import OrderedDict

from .distance import distance
from .text import Embedder


weights = OrderedDict(
    [
        ("skills", 1.0),
        ("user_data", 1.0),
        ("wishes", 1.0),
    ]
)


def get_search_vectors(
    searching_user,  # noqa: ANN001
    candidate_wishes: str = None,
    required_skills: str = None | list[float],
    use_gpu: bool = False,
) -> tuple[list[float] | None, list[float], list[float] | None]:
    """Get 3 vectors for searching system.

    :param searching_user: User searching candidates.
    :param candidate_wishes: Wishes for candidate (just text). Default = None.
    :param required_skills: Candidate's hard skills (can be embedding).
    :param use_gpu: 'True' if you need to use gpu.
    :return: Several vectors for search system.
    """
    user_data = [searching_user.gender, searching_user.age]
    text_embedder = Embedder(use_gpu=use_gpu)

    embedded_wishes = (
        text_embedder.get_text_embedding(candidate_wishes)
        if candidate_wishes is not None
        else None
    )

    if isinstance(required_skills, str):
        embedded_skills = text_embedder.get_text_embedding(required_skills)
    else:
        embedded_skills = required_skills

    return embedded_skills, user_data, embedded_wishes


def comparing_func(
    searching_vectors: tuple[list[float] | None, list[float], list[float] | None],
    candidate_resume,  # noqa: ANN001
    distance_metric: str = "manhattan",
) -> float:
    """Get comparing value of user's searching vectors and candidate's resume.

    :param searching_vectors: Vectors received from get_search_vector function.
    :param candidate_resume: Candidate's Resume.
    :param distance_metric: Type of distance metric.
    """
    skills_compare = 0
    wishes_compare = 0
    embedded_skills, user_data, embedded_wishes = searching_vectors

    user_data_compare = distance(
        a=user_data, b=candidate_resume.user_vector, metric=distance_metric
    )

    if embedded_skills is not None:
        skills_compare = distance(
            a=embedded_skills,
            b=candidate_resume.recommender_vector,
            metric=distance_metric,
        )

    if embedded_wishes is not None:
        wishes_compare = distance(
            a=embedded_wishes,
            b=candidate_resume.employer_wishes,
            metric=distance_metric,
        )

    user_data_compare *= weights["user_data"]
    skills_compare *= weights["skills"]
    wishes_compare *= weights["wishes"]

    return user_data_compare + skills_compare + wishes_compare
