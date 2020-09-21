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
"""Tests for redis_component.executor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random

import apache_beam as beam
from apache_beam.testing import util
import redis
from redis_component import executor
from redis_proto import redis_config_pb2
from redis_proto import redis_hash_query_pb2
import tensorflow as tf
from tfx.proto import example_gen_pb2
from tfx.types import artifact_utils
from tfx.types import standard_artifacts

from google.protobuf import json_format


class ExecutorTestWithMock(tf.test.TestCase):

    def testDeserializeConnConfig(self):
        conn_config = redis_config_pb2.RedisConnConfig()

        deseralized_conn = executor._deserialize_conn_config(conn_config)
        truth_conn = {'decode_responses': True}
        self.assertEqual(truth_conn, deseralized_conn)

    def testRedisToExample(self):
        redis_conn = redis.Redis()
        redis_conn.hmset('test_record', mapping={'i': '1', 'f': '2.0', 's': 'abc'})
        with beam.Pipeline() as pipeline:
            examples = (
                    pipeline | 'ToTFExample' >> executor._RedisToExample(
                exec_properties={
                    'input_config':
                        json_format.MessageToJson(
                            example_gen_pb2.Input(),
                            preserving_proto_field_name=True),
                    'custom_config':
                        json_format.MessageToJson(
                            example_gen_pb2.CustomConfig(),
                            preserving_proto_field_name=True)
                },
                split_pattern=json_format.MessageToJson(
                    redis_hash_query_pb2.RedisHashQuery(hash_key_pattern='test_record*', schema=[
                        {
                            'name': 'i',
                            'type': 'integer'
                        },
                        {
                            'name': 'f',
                            'type': 'float'
                        },
                        {
                            'name': 's',
                            'type': 'string'
                        }
                    ]), preserving_proto_field_name=True, including_default_value_fields=True)))

            feature = {}
            feature['i'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[1]))
            feature['f'] = tf.train.Feature(
                float_list=tf.train.FloatList(value=[2.0]))
            feature['s'] = tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes('abc')]))
            example_proto = tf.train.Example(
                features=tf.train.Features(feature=feature))
            print("examples = " + str(examples))
            util.assert_that(examples, util.equal_to([example_proto]))

    def testDo(self):
        output_data_dir = os.path.join(
            os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
            self._testMethodName)

        # Create output dict.
        examples = standard_artifacts.Examples()
        examples.uri = output_data_dir
        output_dict = {'examples': [examples]}

        # Create exe properties.
        exec_properties = {
            'input_config':
                json_format.MessageToJson(
                    example_gen_pb2.Input(splits=[
                        example_gen_pb2.Input.Split(
                            name='bq', pattern=json_format.MessageToJson(
                                redis_hash_query_pb2.RedisHashQuery(hash_key_pattern='record_*', schema=[
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
                                ]))),
                    ]),
                    preserving_proto_field_name=True, including_default_value_fields=True,),
            'custom_config':
                json_format.MessageToJson(example_gen_pb2.CustomConfig()),
            'output_config':
                json_format.MessageToJson(
                    example_gen_pb2.Output(
                        split_config=example_gen_pb2.SplitConfig(splits=[
                            example_gen_pb2.SplitConfig.Split(
                                name='train', hash_buckets=2),
                            example_gen_pb2.SplitConfig.Split(
                                name='eval', hash_buckets=1)
                        ]))),
        }

        # Run executor.
        redis_example_gen = executor.Executor()
        redis_example_gen.Do({}, output_dict, exec_properties)

        self.assertEqual(
            artifact_utils.encode_split_names(['train', 'eval']),
            examples.split_names)

        # Check Redis example gen outputs.
        train_output_file = os.path.join(examples.uri, 'train',
                                         'data_tfrecord-00000-of-00001.gz')
        eval_output_file = os.path.join(examples.uri, 'eval',
                                        'data_tfrecord-00000-of-00001.gz')
        self.assertTrue(tf.io.gfile.exists(train_output_file))
        self.assertTrue(tf.io.gfile.exists(eval_output_file))
        self.assertGreater(
            tf.io.gfile.GFile(train_output_file).size(),
            tf.io.gfile.GFile(eval_output_file).size())


if __name__ == '__main__':
    tf.test.main()
