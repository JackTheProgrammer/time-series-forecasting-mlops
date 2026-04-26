from pydantic import BaseModel, ConfigDict
from pandas import DataFrame
from torch import Tensor

class ForecastRequest(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    series_frame: DataFrame
    
class ForecastResponse(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    forecasted_price: Tensor