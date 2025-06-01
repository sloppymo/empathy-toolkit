#!/usr/bin/env python3
"""
Empathy Toolkit Test Script

This script tests the major components of the empathy toolkit to ensure they are working correctly.
It checks dataset processing, basic model operations, and utility functions.
"""

import os
import sys
import json
import tempfile
import unittest
from pathlib import Path

# Add the current directory to the path so we can import the modules
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

class EmpathyToolkitTest(unittest.TestCase):
    """Test suite for the Empathy Toolkit components"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_dir = tempfile.mkdtemp()
        
        # Create a small test dataset
        self.test_dataset_path = os.path.join(self.test_dir, "test_dataset.jsonl")
        with open(self.test_dataset_path, "w", encoding="utf-8") as f:
            f.write('{"prompt": "I feel really sad today.", "response": "I understand that you\'re feeling sad. It\'s completely normal to have days like this. Would you like to talk about what\'s causing these feelings?"}\n')
            f.write('{"prompt": "I just got promoted at work!", "response": "That\'s wonderful news! Congratulations on your promotion. Your hard work and dedication have paid off. How are you celebrating this achievement?"}\n')
            f.write('{"prompt": "I\'m worried about my upcoming exam.", "response": "It\'s natural to feel anxious about exams. Many people experience test anxiety. Have you tried any relaxation techniques or study strategies that have helped you in the past?"}\n')
    
    def tearDown(self):
        """Clean up test environment"""
        # Remove test files
        if os.path.exists(self.test_dataset_path):
            os.remove(self.test_dataset_path)
        
        # Remove any other test files created during tests
        for file in os.listdir(self.test_dir):
            os.remove(os.path.join(self.test_dir, file))
        
        # Remove test directory
        os.rmdir(self.test_dir)
    
    def test_dataset_conversion(self):
        """Test the dataset conversion functionality"""
        try:
            from dataset_processing.convert_dataset import convert_dataset, validate_dataset
            
            # Convert the test dataset
            output_path = os.path.join(self.test_dir, "converted_dataset.jsonl")
            convert_dataset(self.test_dataset_path, output_path)
            
            # Verify the converted dataset exists
            self.assertTrue(os.path.exists(output_path), "Converted dataset file was not created")
            
            # Verify the converted dataset has the correct format
            with open(output_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                self.assertEqual(len(lines), 3, "Converted dataset should have 3 entries")
                
                # Check the format of the first entry
                entry = json.loads(lines[0])
                self.assertIn("messages", entry, "Entry should have 'messages' field")
                self.assertTrue(isinstance(entry["messages"], list), "'messages' should be a list")
                self.assertGreaterEqual(len(entry["messages"]), 2, "There should be at least 2 messages")
            
            # Validate the dataset
            validation_result = validate_dataset(output_path)
            self.assertTrue(validation_result, "Dataset validation failed")
            
            print("[PASS] Dataset conversion test passed")
            
        except ImportError as e:
            self.fail(f"Module import failed: {e}")
        except Exception as e:
            self.fail(f"Dataset conversion test failed: {e}")
    
    def test_empathy_evaluation(self):
        """Test the empathy evaluation functionality"""
        try:
            # Create a test response
            test_response = "I understand that you're feeling sad. It's completely normal to have days like this. I'm here to listen if you want to talk about what's causing these feelings. Remember to be kind to yourself during difficult times."
            
            # Import the ResponseEvaluator class
            sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "api"))
            from api.empathy_api_service import ResponseEvaluator
            
            # Create a response evaluator
            evaluator = ResponseEvaluator()
            
            # Evaluate the test response
            metrics = evaluator.count_empathy_indicators(test_response)
            
            # Verify metrics are returned
            self.assertIsNotNone(metrics, "Metrics should not be None")
            self.assertTrue(isinstance(metrics, dict), "Metrics should be a dictionary")
            self.assertIn("validation", metrics, "Metrics should include validation count")
            self.assertIn("reflection", metrics, "Metrics should include reflection count")
            self.assertIn("support", metrics, "Metrics should include support count")
            self.assertIn("total", metrics, "Metrics should include total count")
            
            # Verify that at least one empathy indicator was found
            self.assertGreater(metrics["total"], 0, "No empathy indicators found in a clearly empathetic response")
            
            print("[PASS] Empathy evaluation test passed")
            
        except ImportError as e:
            self.fail(f"Module import failed: {e}")
        except Exception as e:
            self.fail(f"Empathy evaluation test failed: {e}")
    
    def test_file_structure(self):
        """Test that the file structure of the toolkit is correct"""
        # Define expected directories
        expected_dirs = [
            "dataset_processing",
            "model_training",
            "analysis",
            "utilities",
            "applications",
            "api"
        ]
        
        # Check that each directory exists
        for dir_name in expected_dirs:
            dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), dir_name)
            self.assertTrue(os.path.isdir(dir_path), f"Directory {dir_name} does not exist")
            
            # Check that each directory has an __init__.py file
            init_path = os.path.join(dir_path, "__init__.py")
            self.assertTrue(os.path.exists(init_path), f"Directory {dir_name} does not have an __init__.py file")
        
        print("[PASS] File structure test passed")

    def test_requirements(self):
        """Test that the requirements.txt file exists and has necessary packages"""
        requirements_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "requirements.txt")
        self.assertTrue(os.path.exists(requirements_path), "requirements.txt file does not exist")
        
        # Check that requirements.txt has content
        with open(requirements_path, "r", encoding="utf-8") as f:
            content = f.read()
            self.assertGreater(len(content), 0, "requirements.txt file is empty")
            
            # Check for essential packages
            essential_packages = ["numpy", "pandas", "flask", "httpx"]
            for package in essential_packages:
                self.assertIn(package, content, f"requirements.txt is missing {package}")
        
        print("[PASS] Requirements test passed")

def run_tests():
    """Run all tests"""
    # Create and run the test suite
    test_suite = unittest.TestLoader().loadTestsFromTestCase(EmpathyToolkitTest)
    test_result = unittest.TextTestRunner(verbosity=2).run(test_suite)
    
    # Print summary
    print("\n=== Empathy Toolkit Test Summary ===")
    print(f"Tests run: {test_result.testsRun}")
    print(f"Failures: {len(test_result.failures)}")
    print(f"Errors: {len(test_result.errors)}")
    
    if not test_result.failures and not test_result.errors:
        print("\n[SUCCESS] All tests passed! The Empathy Toolkit is working correctly.")
    else:
        print("\n[FAILED] Some tests failed. Please check the errors above.")
    
    # Return True if all tests passed, False otherwise
    return len(test_result.failures) == 0 and len(test_result.errors) == 0

if __name__ == "__main__":
    run_tests()
