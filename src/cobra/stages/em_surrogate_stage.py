from cobra.stages.base_stage import COBRABaseStage


class EMSurrogateStage(COBRABaseStage):
    """
    EM Surrogate Stage - This stage performs the EM simulation using a surrogate model.
    It takes the current design state, runs the surrogate model, and updates the design state with the new EM results.
    """

    def __init__(self, em_surrogate_model):
        self.em_surrogate_model = em_surrogate_model

    def run(self, context):
        pass
