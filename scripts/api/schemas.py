# from typing import TypedDict
from torch import Tensor
from pydantic import BaseModel, ConfigDict
# from typing import Annotated
from pydantic_core import core_schema

class TorchTensorType:
    @classmethod
    def __get_pydantic_core_schema__(cls, _source_type, _handler):
        return core_schema.json_or_python_schema(
            json_schema=core_schema.list_schema(),
            python_schema=core_schema.is_instance_schema(Tensor),
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda t: t.tolist()
            ),
        )

class ForecastRequest(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, from_attributes=True)
    forecast_series: list
    
class ForecastResponse(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    forecasted_price: TorchTensorType
    
# forecast_res = ForecastResponse(forecasted_price=Tensor([1.0, 2.0, 3.0]))
# print(forecast_res.model_dump_json())