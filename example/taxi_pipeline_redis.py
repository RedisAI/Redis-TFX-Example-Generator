# Lint as: python2, python3
# Copyright 2019 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Chicago taxi example using TFX."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from typing import Text

import absl
from redis_component.component import RedisExampleGen
from redis_proto import redis_config_pb2
from redis_proto import redis_hash_query_pb2

from tfx.components import Evaluator
from tfx.components import ExampleValidator
from tfx.components import ModelValidator
from tfx.components import Pusher
from tfx.components import SchemaGen
from tfx.components import StatisticsGen
from tfx.components import Trainer
from tfx.components import Transform
from tfx.orchestration import metadata
from tfx.orchestration import pipeline
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner
from tfx.proto import evaluator_pb2
from tfx.proto import pusher_pb2
from tfx.proto import trainer_pb2

_pipeline_name = 'chicago_taxi_redis'

# This example assumes that the taxi data is stored in ~/taxi/data and the
# taxi utility function is in ~/taxi.  Feel free to customize this as needed.
_taxi_root = os.path.join(os.environ['HOME'], 'taxi')
# Redis configuration that corresponds with tutorial in README.md
_redis_config = redis_config_pb2.RedisConnConfig(host="localhost")
# The query that extracts the Chicago taxi data examples from Redis, following
# setup as described in the README.md
_pattern = 'record_*'
_schema = [
    {
        'name': 'pickup_community_area',
        'type': 'integer'
    },
    {
        'name': 'fare',
        'type': 'float'
    },
    {
        'name': 'trip_start_month',
        'type': 'integer'
    },
    {
        'name': 'trip_start_hour',
        'type': 'integer'
    },
    {
        'name': 'trip_start_day',
        'type': 'integer'
    },
    {
        'name': 'trip_start_timestamp',
        'type': 'integer'
    },
    {
        'name': 'pickup_latitude',
        'type': 'float'
    },
    {
        'name': 'pickup_longitude',
        'type': 'float'
    },
    {
        'name': 'dropoff_latitude',
        'type': 'float'
    },
    {
        'name': 'dropoff_longitude',
        'type': 'float'
    },
    {
        'name': 'trip_miles',
        'type': 'float'
    },
    {
        'name': 'pickup_census_tract',
        'type': 'integer'
    },
    {
        'name': 'dropoff_census_tract',
        'type': 'float'
    },
    {
        'name': 'payment_type',
        'type': 'string'
    },
    {
        'name': 'company',
        'type': 'string'
    },
    {
        'name': 'trip_seconds',
        'type': 'float'
    },
    {
        'name': 'dropoff_community_area',
        'type': 'float'
    },
    {
        'name': 'tips',
        'type': 'float'
    },
    {
        'name': 'big_tipper',
        'type': 'integer'
    }
]
_query = redis_hash_query_pb2.RedisHashQuery(hash_key_pattern=_pattern, schema=_schema)
# Python module file to inject customized logic into the TFX components. The
# Transform and Trainer both require user-defined functions to run successfully.
_module_file = os.path.join('taxi_utils.py')
# Path which can be listened to by the model server.  Pusher will output the
# trained model here.
_serving_model_dir = os.path.join(_taxi_root, 'serving_model', _pipeline_name)

# Directory and data locations.  This example assumes all of the chicago taxi
# example code and metadata library is relative to $HOME, but you can store
# these files anywhere on your local filesystem.
_tfx_root = os.path.join(os.environ['HOME'], 'tfx')
_pipeline_root = os.path.join(_tfx_root, 'pipelines', _pipeline_name)
# Sqlite ML-metadata db path.
_metadata_path = os.path.join(_tfx_root, 'metadata', _pipeline_name,
                              'metadata.db')


def _create_pipeline(pipeline_name: Text, pipeline_root: Text,
                     module_file: Text,
                     redis_config: redis_config_pb2.RedisConnConfig,
                     query: redis_hash_query_pb2, serving_model_dir: Text,
                     metadata_path: Text) -> pipeline.Pipeline:
    """Implements the chicago taxi pipeline with TFX."""
    # Brings data into the pipeline or otherwise joins/converts training data
    example_gen = RedisExampleGen(conn_config=redis_config, query=query)

    # Computes statistics over data for visualization and example validation.
    statistics_gen = StatisticsGen(examples=example_gen.outputs['examples'])

    # Generates schema based on statistics files.
    schema_gen = SchemaGen(statistics=statistics_gen.outputs['statistics'])

    # Performs anomaly detection based on statistics and data schema.
    example_validator = ExampleValidator(
        statistics=statistics_gen.outputs['statistics'],
        schema=schema_gen.outputs['schema'])

    # Performs transformations and feature engineering in training and serving.
    transform = Transform(
        examples=example_gen.outputs['examples'],
        schema=schema_gen.outputs['schema'],
        module_file=module_file)

    # Uses user-provided Python function that implements a model using TF-Learn.
    trainer = Trainer(
        module_file=module_file,
        transformed_examples=transform.outputs['transformed_examples'],
        schema=schema_gen.outputs['schema'],
        transform_graph=transform.outputs['transform_graph'],
        train_args=trainer_pb2.TrainArgs(num_steps=10000),
        eval_args=trainer_pb2.EvalArgs(num_steps=5000))

    # Uses TFMA to compute a evaluation statistics over features of a model.
    evaluator = Evaluator(
        examples=example_gen.outputs['examples'],
        model=trainer.outputs['model'],
        feature_slicing_spec=evaluator_pb2.FeatureSlicingSpec(specs=[
            evaluator_pb2.SingleSlicingSpec(
                column_for_slicing=['trip_start_hour'])
        ]))

    # Performs quality validation of a candidate model (compared to a baseline).
    model_validator = ModelValidator(
        examples=example_gen.outputs['examples'], model=trainer.outputs['model'])

    # Checks whether the model passed the validation steps and pushes the model
    # to a file destination if check passed.
    pusher = Pusher(
        model=trainer.outputs['model'],
        model_blessing=model_validator.outputs['blessing'],
        push_destination=pusher_pb2.PushDestination(
            filesystem=pusher_pb2.PushDestination.Filesystem(
                base_directory=serving_model_dir)))

    return pipeline.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        components=[
            example_gen, statistics_gen, schema_gen, example_validator, transform,
            trainer, evaluator, model_validator, pusher
        ],
        enable_cache=False,
        metadata_connection_config=metadata.sqlite_metadata_connection_config(
            metadata_path),
    )


# To run this pipeline from the python CLI:
#   $python taxi_pipeline_redis.py
if __name__ == '__main__':
    absl.logging.set_verbosity(absl.logging.INFO)

    BeamDagRunner().run(
        _create_pipeline(
            pipeline_name=_pipeline_name,
            pipeline_root=_pipeline_root,
            redis_config=_redis_config,
            query=_query,
            module_file=_module_file,
            serving_model_dir=_serving_model_dir,
            metadata_path=_metadata_path))
