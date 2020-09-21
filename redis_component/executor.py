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
"""Generic TFX RedisExampleGen executor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
from typing import Any, Dict, Iterable, Text, Tuple

import apache_beam as beam
import redis
from redis_proto import redis_config_pb2
from redis_proto import redis_hash_query_pb2
import tensorflow as tf
from tfx.components.example_gen import base_example_gen_executor
from tfx.proto import example_gen_pb2

from google.protobuf import json_format


@beam.typehints.with_input_types(Text)
@beam.typehints.with_output_types(beam.typehints.Iterable[Tuple[Text, Text, Text]])
class _ReadRedisDoFn(beam.DoFn):
    """Beam DoFn class that reads from Redis.

    Attributes:
      redis_conn: A Redis client connection.
    """

    def __init__(self, redis_config: Dict):
        self.redis_config = redis_config

    def hash_iter(self, pattern):
        for result in self.redis_conn.scan_iter(match=pattern, _type='HASH'):
            yield self.redis_conn.hgetall(result)

    def process(self, query: Text) -> Iterable[Tuple[Text, Text, Text]]:
        # """Yields rows from Redis pattern scan results.
        #
        # Args:
        #   pattern: The pattern of the keys where each data point is stored.
        #
        # Yields:
        #   One row from the query result, represented by a list of tuples. Each tuple
        #   contains information on column name, column data type, data.
        # """
        self.redis_conn = redis.Redis(**(self.redis_config))
        cols = []
        col_types = []
        query_pb = redis_hash_query_pb2.RedisHashQuery()
        query_pb = json_format.Parse(query, query_pb)
        for i in range(len(query_pb.schema)):
            pair_schema = query_pb.schema[i]
            cols.append(pair_schema.name)
            col_types.append(redis_hash_query_pb2.redis_hash_pair_schema.hash_field_type.Name(pair_schema.type))
        for result in self.hash_iter(query_pb.hash_key_pattern):
            values = [result.get(col_name) for col_name in cols]
            yield list(zip(cols, col_types, values))

    def teardown(self):
        if self.redis_conn:
            self.redis_conn.close()


def _deserialize_conn_config(conn_config: redis_config_pb2.RedisConnConfig) -> Dict:
    """Deserializes Redis connection config to Redis python client.

    Args:
      conn_config: Protobuf-encoded connection config for Redis client.

    Returns:
      A redis.Redis instance initialized with user-supplied
      parameters.
    """
    params = {'decode_responses': True}
    # Only deserialize rest of parameters if set by user
    if conn_config.HasField('host'):
        params['host'] = conn_config.host
    if conn_config.HasField('port'):
        params['port'] = conn_config.port
    if conn_config.HasField('username'):
        params['username'] = conn_config.username
    if conn_config.HasField('password'):
        params['password'] = conn_config.password
    if conn_config.HasField('db'):
        params['db'] = conn_config.db
    return params


def _row_to_example(instance: Iterable[Tuple[Text, Text, Text]]) -> tf.train.Example:
    """Convert Redis result row to tf example."""
    feature = {}
    for key, data_type, value in instance:
        if value is None:
            feature[key] = tf.train.Feature()
            continue
        elif data_type == 'integer':
            feature[key] = tf.train.Feature(
                int64_list=tf.train.Int64List(value=[int(float(value)) if value != 'nan' else 0]))
        elif data_type == 'float':
            feature[key] = tf.train.Feature(
                float_list=tf.train.FloatList(value=[float(value) if value != 'nan' else 0.0]))
        elif data_type == 'string':
            feature[key] = tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(value if value != 'nan' else '')]))
        else:
            raise RuntimeError(
                'Column type {} is not supported.'.format(data_type))
    return tf.train.Example(features=tf.train.Features(feature=feature))


@beam.ptransform_fn
@beam.typehints.with_input_types(beam.Pipeline)
@beam.typehints.with_output_types(tf.train.Example)
def _RedisToExample(  # pylint: disable=invalid-name
        pipeline: beam.Pipeline,
        exec_properties: Dict[Text, Any],
        split_pattern: Text) -> beam.pvalue.PCollection:
    """Read from Redis and transform to TF examples.

    Args:
      pipeline: beam pipeline.
      exec_properties: A dict of execution properties.
      split_pattern: Split.pattern in Input config, a Redis keys pattern string.

    Returns:
      PCollection of TF examples.
    """
    conn_config = example_gen_pb2.CustomConfig()
    json_format.Parse(exec_properties['custom_config'], conn_config)
    redis_config = redis_config_pb2.RedisConnConfig()
    conn_config.custom_config.Unpack(redis_config)

    client_config = _deserialize_conn_config(redis_config)
    return (pipeline
            | 'Query' >> beam.Create([split_pattern])
            | 'QueryRedis' >> beam.ParDo(_ReadRedisDoFn(client_config))
            | 'ToTFExample' >> beam.Map(_row_to_example))


class Executor(base_example_gen_executor.BaseExampleGenExecutor):
    """Generic TFX RedisExampleGen executor."""

    def GetInputSourceToExamplePTransform(self) -> beam.PTransform:
        """Returns PTransform for Redis to TF examples."""
        return _RedisToExample
