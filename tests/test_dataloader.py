# tests/test_dataloader.py

# tests/test_dataloader.py

import unittest
import os
import sys
import numpy as np  # Added numpy import
sys.path.append('..')  # Adjust the path as needed
from dataloader import DataLoader
from dataloader.utils import timer
from dataloader.preprocessors import default_preprocess, augment_image, augment_text
from collections import namedtuple

class TestDataLoader(unittest.TestCase):

    def test_dataloader_initialization(self):
        """Test Case 1: DataLoader Class Initialization and Data Downloading"""
        data_loader = DataLoader(dataset_name='MNIST', batch_size=32)
        self.assertIsInstance(data_loader, DataLoader)
        self.assertEqual(data_loader.dataset_name, 'MNIST')
        self.assertTrue(os.path.exists('datasets/MNIST'))

    def test_flexible_method_parameters(self):
        """Test Case 2: Flexible Method Parameters"""
        def custom_preprocess(sample, factor=2):
            return sample * factor

        data_loader = DataLoader(preprocess_func=custom_preprocess)
        self.assertTrue(callable(data_loader.kwargs['preprocess_func']))

    def test_data_normalization(self):
        """Test Case 3: Data Normalization"""
        from dataloader.preprocessors import normalize

        sample_data = [0, 128, 255]
        normalized_data = list(map(normalize, sample_data))
        for value in normalized_data:
            self.assertTrue(0.0 <= value <= 1.0)

    def test_lambda_functions(self):
        """Test Case 4: Use of Lambda Functions"""
        data = [1, 2, 3, 4, 5]
        transformed_data = list(map(lambda x: x * 2, data))
        self.assertEqual(transformed_data, [2, 4, 6, 8, 10])

    def test_closures(self):
        """Test Case 5: Implementation of Closures"""
        def make_multiplier(factor):
            def multiply(number):
                return number * factor
            return multiply

        times_three = make_multiplier(3)
        self.assertEqual(times_three(10), 30)

    def test_decorators(self):
        """Test Case 6: Use of Decorators for Logging"""
        @timer
        def sample_function():
            return True

        result = sample_function()
        self.assertTrue(result)

    def test_namedtuples(self):
        """Test Case 7: Use of NamedTuples"""
        DataSample = namedtuple('DataSample', ['features', 'label'])
        sample = DataSample(features=[1, 2, 3], label=0)
        self.assertEqual(sample.features, [1, 2, 3])
        self.assertEqual(sample.label, 0)

    def test_project_modularization(self):
        """Test Case 8: Proper Project Modularization"""
        # Check if modules can be imported
        try:
            from dataloader import dataloader
            from dataloader import preprocessors
            from dataloader import utils
        except ImportError:
            self.fail("Modules not properly organized")

    def test_fstrings(self):
        """Test Case 9: Use of f-Strings"""
        name = 'DataLoader'
        message = f"Initializing {name}"
        self.assertEqual(message, "Initializing DataLoader")

    def test_custom_iterator(self):
        """Test Case 10: Implementation of Custom Iterator"""
        data_loader = DataLoader(batch_size=10)
        iterator = iter(data_loader)
        batch = next(iterator)
        self.assertEqual(len(batch), 10)

    def test_generators(self):
        """Test Case 11: Use of Generators for Lazy Loading"""
        def data_generator():
            for i in range(10):
                yield i

        gen = data_generator()
        self.assertEqual(next(gen), 0)
        self.assertEqual(next(gen), 1)

    def test_list_comprehensions(self):
        """Test Case 12: Use of List Comprehensions"""
        data = [i for i in range(5)]
        self.assertEqual(data, [0, 1, 2, 3, 4])

    def test_context_managers(self):
        """Test Case 13: Use of Context Managers"""
        try:
            with open('testfile.txt', 'w') as f:
                f.write('Test')
            self.assertTrue(os.path.exists('testfile.txt'))
        finally:
            if os.path.exists('testfile.txt'):
                os.remove('testfile.txt')

    def test_exception_handling(self):
        """Test Case 14: Exception Handling"""
        try:
            x = 1 / 0
        except ZeroDivisionError:
            self.assertTrue(True)
        else:
            self.fail("ZeroDivisionError not raised")

    def test_command_line_arguments(self):
        """Test Case 15: Command-Line Arguments"""
        sys.argv = ['main.py', 'CIFAR-10', '64']
        from main import main
        try:
            main()
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"main() raised Exception unexpectedly: {e}")

    def test_csv_data_loading(self):
        """Test Case 16: CSV Data Loading"""
        data_loader = DataLoader(dataset_name='nyc_csv_data', batch_size=10)
        self.assertIsInstance(data_loader, DataLoader)
        self.assertTrue(os.path.exists('datasets/nyc_csv_data/nyc_parking_tickets_extract-1.csv'))

    def test_text_data_loading(self):
        """Test Case 17: Text Data Loading"""
        data_loader = DataLoader(dataset_name='smsspamcollection', batch_size=10)
        self.assertIsInstance(data_loader, DataLoader)
        self.assertTrue(os.path.exists('datasets/smsspamcollection/SMSSpamCollection'))

    def test_image_data_augmentation(self):
        """Test Case 18: Image Data Augmentation"""
        image = np.random.rand(32, 32, 3)
        augmented_image = augment_image(image)
        self.assertEqual(augmented_image.shape, image.shape)

    def test_text_data_augmentation(self):
        """Test Case 19: Text Data Augmentation"""
        text = "This is a test sentence."
        augmented_text = augment_text(text)
        self.assertNotEqual(augmented_text, text)

if __name__ == '__main__':
    unittest.main()

# Dev 2 :: Working

# import unittest
# import os
# import sys
# import numpy as np
# sys.path.append('..')  # Adjust the path as needed
# from dataloader import DataLoader
# from dataloader.utils import timer
# from dataloader.preprocessors import default_preprocess, augment_image, augment_text
# from collections import namedtuple

# class TestDataLoader(unittest.TestCase):

#     def test_dataloader_initialization(self):
#         """Test Case 1: DataLoader Class Initialization and Data Downloading"""
#         data_loader = DataLoader(dataset_name='MNIST', batch_size=32)
#         self.assertIsInstance(data_loader, DataLoader)
#         self.assertEqual(data_loader.dataset_name, 'MNIST')
#         self.assertTrue(os.path.exists('datasets/MNIST'))

#     def test_flexible_method_parameters(self):
#         """Test Case 2: Flexible Method Parameters"""
#         def custom_preprocess(sample, factor=2):
#             return sample * factor

#         data_loader = DataLoader(preprocess_func=custom_preprocess)
#         self.assertTrue(callable(data_loader.kwargs['preprocess_func']))

#     def test_data_normalization(self):
#         """Test Case 3: Data Normalization"""
#         from dataloader.preprocessors import normalize

#         sample_data = [0, 128, 255]
#         normalized_data = list(map(normalize, sample_data))
#         for value in normalized_data:
#             self.assertTrue(0.0 <= value <= 1.0)

#     def test_lambda_functions(self):
#         """Test Case 4: Use of Lambda Functions"""
#         data = [1, 2, 3, 4, 5]
#         transformed_data = list(map(lambda x: x * 2, data))
#         self.assertEqual(transformed_data, [2, 4, 6, 8, 10])

#     def test_closures(self):
#         """Test Case 5: Implementation of Closures"""
#         def make_multiplier(factor):
#             def multiply(number):
#                 return number * factor
#             return multiply

#         times_three = make_multiplier(3)
#         self.assertEqual(times_three(10), 30)

#     def test_decorators(self):
#         """Test Case 6: Use of Decorators for Logging"""
#         @timer
#         def sample_function():
#             return True

#         result = sample_function()
#         self.assertTrue(result)

#     def test_namedtuples(self):
#         """Test Case 7: Use of NamedTuples"""
#         DataSample = namedtuple('DataSample', ['features', 'label'])
#         sample = DataSample(features=[1, 2, 3], label=0)
#         self.assertEqual(sample.features, [1, 2, 3])
#         self.assertEqual(sample.label, 0)

#     def test_project_modularization(self):
#         """Test Case 8: Proper Project Modularization"""
#         try:
#             from dataloader import dataloader
#             from dataloader import preprocessors
#             from dataloader import utils
#         except ImportError:
#             self.fail("Modules not properly organized")

#     def test_fstrings(self):
#         """Test Case 9: Use of f-Strings"""
#         name = 'DataLoader'
#         message = f"Initializing {name}"
#         self.assertEqual(message, "Initializing DataLoader")

#     def test_custom_iterator(self):
#         """Test Case 10: Implementation of Custom Iterator"""
#         data_loader = DataLoader(batch_size=10)
#         iterator = iter(data_loader)
#         batch = next(iterator)
#         self.assertEqual(len(batch), 10)

#     def test_generators(self):
#         """Test Case 11: Use of Generators for Lazy Loading"""
#         def data_generator():
#             for i in range(10):
#                 yield i

#         gen = data_generator()
#         self.assertEqual(next(gen), 0)
#         self.assertEqual(next(gen), 1)

#     def test_list_comprehensions(self):
#         """Test Case 12: Use of List Comprehensions"""
#         data = [i for i in range(5)]
#         self.assertEqual(data, [0, 1, 2, 3, 4])

#     def test_context_managers(self):
#         """Test Case 13: Use of Context Managers"""
#         try:
#             with open('testfile.txt', 'w') as f:
#                 f.write('Test')
#             self.assertTrue(os.path.exists('testfile.txt'))
#         finally:
#             if os.path.exists('testfile.txt'):
#                 os.remove('testfile.txt')

#     def test_exception_handling(self):
#         """Test Case 14: Exception Handling"""
#         try:
#             x = 1 / 0
#         except ZeroDivisionError:
#             self.assertTrue(True)
#         else:
#             self.fail("ZeroDivisionError not raised")

#     def test_command_line_arguments(self):
#         """Test Case 15: Command-Line Arguments"""
#         sys.argv = ['main.py', 'CIFAR-10', '64']
#         from main import main
#         try:
#             main()
#             self.assertTrue(True)
#         except Exception as e:
#             self.fail(f"main() raised Exception unexpectedly: {e}")

#     def test_csv_data_loading(self):
#         """Test Case 16: CSV Data Loading"""
#         data_loader = DataLoader(dataset_name='nyc_parking_tickets_extract-1', batch_size=10)
#         self.assertIsInstance(data_loader, DataLoader)
#         self.assertTrue(os.path.exists('datasets/nyc_parking_tickets_extract-1'))

#     def test_text_data_loading(self):
#         """Test Case 17: Text Data Loading"""
#         data_loader = DataLoader(dataset_name='tiny_web_data', batch_size=10)
#         self.assertIsInstance(data_loader, DataLoader)
#         self.assertTrue(os.path.exists('datasets/tiny_web_data'))

#     def test_image_data_augmentation(self):
#         """Test Case 18: Image Data Augmentation"""
#         image = np.random.rand(32, 32, 3)
#         augmented_image = augment_image(image)
#         self.assertEqual(augmented_image.shape, image.shape)

#     def test_text_data_augmentation(self):
#         """Test Case 19: Text Data Augmentation"""
#         text = "This is a test sentence."
#         augmented_text = augment_text(text)
#         self.assertNotEqual(augmented_text, text)

# if __name__ == '__main__':
#     unittest.main()


# Dev 1 :: Working

# # tests/test_dataloader.py
# # python -m unittest tests.test_dataloader
# # python -m unittest tests.test_dataloader.TestDataLoader.test_dataloader_initialization
# import unittest
# import os
# import sys
# sys.path.append('..')  # Adjust the path as needed
# from dataloader import DataLoader
# from dataloader.utils import timer
# from dataloader.preprocessors import default_preprocess
# from collections import namedtuple

# class TestDataLoader(unittest.TestCase):

#     def test_dataloader_initialization(self):
#         """Test Case 1: DataLoader Class Initialization and Data Downloading"""
#         data_loader = DataLoader(dataset_name='MNIST', batch_size=32)
#         self.assertIsInstance(data_loader, DataLoader)
#         self.assertEqual(data_loader.dataset_name, 'MNIST')
#         self.assertTrue(os.path.exists('datasets/MNIST'))

#     def test_flexible_method_parameters(self):
#         """Test Case 2: Flexible Method Parameters"""
#         def custom_preprocess(sample, factor=2):
#             # Example of using positional and keyword arguments
#             return sample * factor

#         data_loader = DataLoader(preprocess_func=custom_preprocess)
#         self.assertTrue(callable(data_loader.kwargs['preprocess_func']))

#     def test_data_normalization(self):
#         """Test Case 3: Data Normalization"""
#         from dataloader.preprocessors import normalize

#         sample_data = [0, 128, 255]
#         normalized_data = list(map(normalize, sample_data))
#         for value in normalized_data:
#             self.assertTrue(0.0 <= value <= 1.0)

#     def test_lambda_functions(self):
#         """Test Case 4: Use of Lambda Functions"""
#         data = [1, 2, 3, 4, 5]
#         transformed_data = list(map(lambda x: x * 2, data))
#         self.assertEqual(transformed_data, [2, 4, 6, 8, 10])

#     def test_closures(self):
#         """Test Case 5: Implementation of Closures"""
#         def make_multiplier(factor):
#             def multiply(number):
#                 return number * factor
#             return multiply

#         times_three = make_multiplier(3)
#         self.assertEqual(times_three(10), 30)

#     def test_decorators(self):
#         """Test Case 6: Use of Decorators for Logging"""
#         @timer
#         def sample_function():
#             return True

#         result = sample_function()
#         self.assertTrue(result)

#     def test_namedtuples(self):
#         """Test Case 7: Use of NamedTuples"""
#         DataSample = namedtuple('DataSample', ['features', 'label'])
#         sample = DataSample(features=[1, 2, 3], label=0)
#         self.assertEqual(sample.features, [1, 2, 3])
#         self.assertEqual(sample.label, 0)

#     def test_project_modularization(self):
#         """Test Case 8: Proper Project Modularization"""
#         # Check if modules can be imported
#         try:
#             from dataloader import dataloader
#             from dataloader import preprocessors
#             from dataloader import utils
#         except ImportError:
#             self.fail("Modules not properly organized")

#     def test_fstrings(self):
#         """Test Case 9: Use of f-Strings"""
#         name = 'DataLoader'
#         message = f"Initializing {name}"
#         self.assertEqual(message, "Initializing DataLoader")

#     def test_custom_iterator(self):
#         """Test Case 10: Implementation of Custom Iterator"""
#         data_loader = DataLoader(batch_size=10)
#         iterator = iter(data_loader)
#         batch = next(iterator)
#         self.assertEqual(len(batch), 10)

#     def test_generators(self):
#         """Test Case 11: Use of Generators for Lazy Loading"""
#         def data_generator():
#             for i in range(10):
#                 yield i

#         gen = data_generator()
#         self.assertEqual(next(gen), 0)
#         self.assertEqual(next(gen), 1)

#     def test_list_comprehensions(self):
#         """Test Case 12: Use of List Comprehensions"""
#         data = [i for i in range(5)]
#         self.assertEqual(data, [0, 1, 2, 3, 4])

#     def test_context_managers(self):
#         """Test Case 13: Use of Context Managers"""
#         try:
#             with open('testfile.txt', 'w') as f:
#                 f.write('Test')
#             self.assertTrue(os.path.exists('testfile.txt'))
#         finally:
#             if os.path.exists('testfile.txt'):
#                 os.remove('testfile.txt')

#     def test_exception_handling(self):
#         """Test Case 14: Exception Handling"""
#         try:
#             x = 1 / 0
#         except ZeroDivisionError:
#             self.assertTrue(True)
#         else:
#             self.fail("ZeroDivisionError not raised")

#     def test_command_line_arguments(self):
#         """Test Case 15: Command-Line Arguments"""
#         # Simulate command-line arguments
#         sys.argv = ['main.py', 'CIFAR-10', '64']
#         from main import main
#         try:
#             main()
#             self.assertTrue(True)
#         except Exception as e:
#             self.fail(f"main() raised Exception unexpectedly: {e}")

# if __name__ == '__main__':
#     unittest.main()
